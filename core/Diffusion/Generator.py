import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from core.AdaptGrad import AdaptGrad
from core.Diffusion.Engine import Engine

from models.Losses import CounterfactualLoss
from uitls.utils import visualize_adaption, visualize_seg


class CounterfactualGenerator:
    def __init__(self,
                 segmentation: nn.Module,
                 diffusion_model: nn.Module,
                 engine: Engine,
                 device: torch.device,
                 cfg: DictConfig,
                 name='counterfactual'):
        super().__init__()
        self.segmentation_model = segmentation
        self.diffusion_model = diffusion_model
        self.engine = engine
        self.device = device
        self.cfg = cfg
        self.diffusion_checkpoint = cfg.diffusion.checkpoint_path
        self.diffusion_name = cfg.diffusion.name
        self.segmentation_checkpoint = cfg.segmentation.checkpoint_path
        self.segmentation_name = cfg.segmentation.name
        self.weight_classifier = cfg.generation.weight_classifier
        self.weight_distance = cfg.generation.weight_distance
        self.adaptgrad = AdaptGrad(cfg)

        # Initialize TensorBoard
        print(f"logging to {cfg.system.log_dir}/Generation/{name}")
        self.writer = SummaryWriter(log_dir=f"{cfg.system.log_dir}/Generation/{name}")

        self.load_checkpoint(f"{self.diffusion_checkpoint}/{self.diffusion_name}.pt",
                             f"{self.segmentation_checkpoint}/{self.segmentation_name}.pt")
        self.segmentation_model.eval()
        self.diffusion_model.eval()

    def load_checkpoint(self, diffusion_checkpoint: str, segmentation_checkpoint: str):
        checkpoint = torch.load(diffusion_checkpoint, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])

        checkpoint = torch.load(segmentation_checkpoint, map_location=self.device)
        self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])

    @torch.no_grad()
    def generate(self, image: Tensor, target: Tensor, start_fraction: float):
        image = image.to(self.device)
        target = target.to(self.device)

        self.diffusion_model.eval()
        self.segmentation_model.eval()

        self.segmentation_model.to(self.device)
        self.diffusion_model.to(self.device)
        self.adaptgrad.to(self.device)

        prediction_org = self.segmentation_model(image).detach().cpu().argmax(dim=1)
        counterfactual = self._generate_counterfactual(image, target, start_fraction)
        adaptions = counterfactual - image
        prediction_cf = self.segmentation_model(counterfactual).detach().cpu().argmax(dim=1)

        self.writer.close()

        return counterfactual.cpu(), adaptions.cpu(), prediction_org.cpu(), prediction_cf.cpu()

    @torch.no_grad()
    def _generate_counterfactual(self,
                                 image: torch.Tensor,
                                 target: torch.Tensor,
                                 start_fraction: float):
        loss_fn = CounterfactualLoss(target.repeat((self.adaptgrad.N, 1, 1)))

        start_t = int(self.engine.T * start_fraction)
        t_tensor = torch.full((image.shape[0],), start_t, device=self.device)
        x, _ = self.engine.diffuse(image, t_tensor)

        for t in tqdm(reversed(range(start_t)), total=start_t, desc="generating counterfactual"):
            t_tensor = torch.full((image.shape[0],), t, device=self.device)
            epsilon_pred = self.diffusion_model(x, t_tensor).detach()
            mean = self.engine.reverse(x, epsilon_pred, t_tensor)

            mu_norm = torch.norm(mean.view(mean.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
            sigma = self.engine.get_sigma(t_tensor, image.shape)

            with torch.enable_grad():
                g_update, approx_image, grad_c, grad_d, logits = self._calc_g_update(
                    x,
                    image,
                    loss_fn,
                    epsilon_pred,
                    t_tensor)

            if t > 0:
                mean = mean - (sigma ** 2 * mu_norm * g_update)
                x = mean + sigma * torch.randn_like(image)
            else:
                x = mean

            # --- TensorBoard Logging ---
            if t % 50 == 0:
                self.log(approx_image,
                         g_update,
                         grad_c,
                         grad_d,
                         logits,
                         mean,
                         mu_norm,
                         start_t,
                         t,
                         x)

        return x.detach()

    def _calc_g_update(self, x: Tensor, image: Tensor, loss_fn: CounterfactualLoss,
                       epsilon_pred: Tensor,
                       t: Tensor):
        x = x.detach().requires_grad_(True)
        approx_image = self.engine.image_approximation(x, epsilon_pred, t)

        # Segmentation Loss
        approx_image_batch = self.adaptgrad(approx_image)
        logits = self.segmentation_model(approx_image_batch)
        loss_c = loss_fn(logits)
        grad_c = torch.autograd.grad(loss_c, x, retain_graph=True)[0]

        grad_c /= (torch.norm(grad_c.view(grad_c.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        # Distance Loss
        distance = torch.norm(image - approx_image, p=2)
        grad_d = torch.autograd.grad(distance, x)[0]
        grad_d /= (torch.norm(grad_d.view(grad_d.shape[0], -1), p=2, dim=1).view(-1, 1, 1,
                                                                                 1) + 1e-8)

        g_update = self.weight_classifier * grad_c + self.weight_distance * grad_d

        return g_update, approx_image, grad_c, grad_d, logits

    def log(self, approx_image: Tensor, g_update: Tensor, grad_c: Tensor, grad_d: Tensor,
            logits: Tensor,
            mean: Tensor, mu_norm: Tensor, start_t: int, t: int, x: Tensor):
        log_step = start_t - t
        self.writer.add_image('Process/x_t', vutils.make_grid(x, normalize=True),
                              log_step)
        self.writer.add_image('Process/approx_x_0',
                              vutils.make_grid(approx_image, normalize=True), log_step)

        self.writer.add_scalar('Gradients/mean', mean.mean().item(), log_step)
        self.writer.add_scalar('Gradients/mu_norm', mu_norm.mean().item(), log_step)
        self.writer.add_scalar('Gradients/g_update', g_update.abs().mean().item(), log_step)
        self.writer.add_scalar('Gradients/grad_c', grad_c.abs().mean().item(), log_step)
        self.writer.add_scalar('Gradients/grad_d', grad_d.abs().mean().item(), log_step)

        prediction = torch.argmax(logits.mean(dim=0, keepdim=True), dim=1)
        prediction = visualize_seg(prediction.detach().cpu())[0].permute(2, 0, 1).float()
        g_visual = visualize_adaption(g_update.sum(dim=1, keepdim=True))[0, 0]
        grad_c_visual = visualize_adaption(grad_c.sum(dim=1, keepdim=True))[0, 0]
        grad_d_visual = visualize_adaption(grad_d.sum(dim=1, keepdim=True))[0, 0]

        self.writer.add_image('Process/g_update',
                              vutils.make_grid(g_visual, normalize=True), log_step)
        self.writer.add_image('Process/grad_c',
                              vutils.make_grid(grad_c_visual, normalize=True), log_step)
        self.writer.add_image('Process/grad_d',
                              vutils.make_grid(grad_d_visual, normalize=True), log_step)
        self.writer.add_image('Process/prediction', prediction, log_step)

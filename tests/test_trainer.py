import pytest
import torch
from core.DiffusionEngine import DiffusionEngine
from core.DiffusionTrainer import DiffusionTrainer
from models.DiffusionModel import DiffusionModel  # Assuming your model class name


@pytest.fixture
def trainer_setup():
    device = torch.device("cpu")  # GitHub Actions friendly
    model = DiffusionModel(1, 32, 32, 128)
    engine = DiffusionEngine(T=10, beta_min=1e-4, beta_max=2e-2, device=device, beta_schedule='linear')

    trainer = DiffusionTrainer(
        model=model,
        engine=engine,
        device=device,
        name="TestRun"
    )
    return trainer


def test_train_step_execution(trainer_setup):
    trainer = trainer_setup
    dummy_images = torch.randn(4, 1, 32, 32)

    # We don't run the full 'train_diffusion' loop.
    # We test the core logic inside the loop manually.
    trainer.model.train()

    # 1. Forward through Engine
    x_t, noise, t = trainer.engine.sample(dummy_images)

    # 2. Forward through Model
    pred = trainer.model(x_t, t)

    # 3. Loss Calculation
    loss = trainer.loss_fn(pred, noise)

    assert loss > 0
    assert not torch.isnan(loss)

    # 4. Backward Pass (Does it crash?)
    loss.backward()
    print("✅ Trainer backprop works!")


def test_visualization_methods(trainer_setup):
    trainer = trainer_setup
    fixed_images = torch.randn(2, 1, 32, 32)

    # We wrap this in a try/except or just run it to see if it throws an error
    try:
        trainer._visualize_one_step(fixed_images, epoch=1)
        # If your generate_image works:
        # trainer._visualize_sampling(epoch=1)
        print("✅ Visualization logic is sound!")
    except Exception as e:
        pytest.fail(f"Visualization failed with error: {e}")
from torchview import draw_graph

from diff_cf.models.diffusion_model import DiffusionModel


def main():
    model = DiffusionModel(
            channels=3,
            width=256,
            height=256,
            time_dim=256,
            nums_features=[128, 256, 512, 1024, 2048],
            use_prior=False,
            num_classes=19,
        )


    # Pass multiple shapes as a list to input_shape

    model_graph = draw_graph(
        model,
        input_size=[(1, 3, 256, 256), (1,)],
        expand_nested=False,  # Groups operations inside ResBlocks/Attention modules
        depth=2  # Only renders top-level U-Net stages
    )

    # Render image file
    model_graph.visual_graph.render("./figures/diffusion_torchview", format="png")
   
if __name__ == '__main__':
    main()

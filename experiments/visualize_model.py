import visualtorch

from diff_cf.models.diffusion_model_final import DiffusionModel


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
    img = visualtorch.render(
        model,
        input_shape=((1, 3, 256, 256), (1,)),
        style="graph",  # Options: "graph", "flow", or "lenet"
        legend=True
    )

    img.save("./figures/diffusion_visualtorch.png")
   
if __name__ == '__main__':
    main()

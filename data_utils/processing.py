from pathlib import Path

import click
import numpy as np
from utils.aggregation import aggregate_data


@click.command()
@click.option(
    "-dataset-path",
    "--dataset-path",
    required=True,
    type=click.Path(exists=True),
    help="dataset_path",
)
@click.option(
    "--threshold",
    "-threshold",
    default=0.9,
    help="Dawidskene threshold",
    show_default=True,
)
def processing(dataset_path: str, threshold: float) -> None:
    """
    processing raw data for training
    """
    if threshold > 1 or threshold < 0:
        raise AttributeError

    np.seterr(divide="ignore")

    public_data = Path(dataset_path)
    result_dir = public_data / f"processed_dataset_{int(threshold*100)}"

    path_names = ["train", "aggregated_dataset", "test"]
    for path_name in path_names:
        (result_dir / path_name).mkdir(parents=True, exist_ok=True)

    aggregate_data(
        data_path=public_data, out_path=result_dir, dawidskene_threshold=threshold
    )


if __name__ == "__main__":
    processing()  # pylint: disable=no-value-for-parameter

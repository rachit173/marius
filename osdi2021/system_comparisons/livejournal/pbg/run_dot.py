import argparse
import random
from pathlib import Path

import attr
import pkg_resources
from torchbiggraph.config import ConfigFileLoader, add_to_sys_path
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.converters.utils import download_url, extract_gzip
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train
from torchbiggraph.util import (
    SubprocessInitializer,
    set_logging_verbosity,
    setup_logging,
)

URL = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
TRAIN_FILENAME = "lj_train.txt"
VALID_FILENAME = "lj_valid.txt"
TEST_FILENAME = "lj_valid.txt"
FILENAMES = [TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME]

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Example on Livejournal")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    parser.add_argument(
        "--data_dir", type=Path, default="data", help="where to save processed data"
    )

    args = parser.parse_args()

    # download data
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    fpath = download_url(URL, data_dir)
    fpath = extract_gzip(fpath)
    print("Downloaded and extracted file.")

    # random split file for train and test
    loader = ConfigFileLoader()
    config = loader.load_config(args.config, args.param)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)
    input_edge_paths = [data_dir / name for name in FILENAMES]
    output_train_path, output_valid_path, output_test_path = config.edge_paths

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=1, rel_col=None),
        dynamic_relations=config.dynamic_relations,
    )

    train_config = attr.evolve(config, edge_paths=[output_train_path])
    train(train_config, subprocess_init=subprocess_init)

    eval_config = attr.evolve(config, edge_paths=[output_test_path])
    do_eval(eval_config, subprocess_init=subprocess_init)


if __name__ == "__main__":
    main()
from .tokenize import UnitDS
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(prog="python -m zachakit.pt")
    subparsers = parser.add_subparsers(dest="sub_module")

    parser_tokenize = subparsers.add_parser("tokenize", help="Faster tokenization")
    parser_tokenize.add_argument("root", type=str, help="root file/dir/repo_id for your dataset")
    parser_tokenize.add_argument("--local", action="store_true", help="whether the root is a local file")
    parser_tokenize.add_argument("--dtype", type=str, choices=["text", "arrow", "csv", "parquet"], default="arrow")
    parser_tokenize.add_argument("--text-col", type=str, help="target text column name in the dataset")
    parser_tokenize.add_argument("--tokenizer", help="tokenizer path/repo_id")
    parser_tokenize.add_argument("--block-size", type=int, default=1024, help="size of blocks to be chunked")
    # FIXME: change default after testing
    parser_tokenize.add_argument("--cache-dir", type=str, help="cache directory for dataset", default="draft/cache")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sub_module == "tokenize":
        uds = UnitDS(
            cache_dir=args.cache_dir,
            from_local_file=args.local,
            text_col=args.text_col,
            root=args.root,
            data_type=args.dtype,
            block_size=args.block_size,
            tokenizer_name=args.tokenizer,
        )

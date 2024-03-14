from .query import QueryManager
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser("python -m zachakit.query")
    parser.add_argument(
        "-i", "--input-dir", required=True, type=str, help="Directory containing json line files used for query."
    )
    parser.add_argument("-o", "--output-dir", required=True, type=str, help="Directory to save result json line files.")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo-1106", help="OpenAI model name used for query.")
    parser.add_argument(
        "--client",
        type=str,
        choices=["local", "azure"],
        default="local",
        help="Use web OpenAI API(local) or use Microsoft Azure API(azure).",
    )
    parser.add_argument("--chunk-size", type=int, default=20, help="Chunk size to save while querying.")
    parser.add_argument(
        "--prompt-debug",
        action="store_true",
        help="Set to estimate prompt tokens. NOTICE: completion tokens cannot be estimated.",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=10,
        help="When such numbers of failure samples are detected, the query process will be shut down.",
    )
    parser.add_argument("--recover_only", action="store_true", help="Recover from ckpt only instead of sending new queries.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qm = QueryManager(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        name=args.model_name,
        client=args.client,
        chunk_size=args.chunk_size,
        prompt_debug=args.prompt_debug,
        failure_limit=args.failure_limit,
    )
    if not args.recover_only:
        qm.live_display()
    qm.recover_result()

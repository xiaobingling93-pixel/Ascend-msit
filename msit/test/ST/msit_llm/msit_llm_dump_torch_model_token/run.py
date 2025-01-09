from main_torch import check_min_version, override_topp_and_topk, parse_args, get_model, precision, performance, \
    cli_demo, webUI


def main():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version('4.30.2')
    override_topp_and_topk()
    args = parse_args()
    tokenizer, model = get_model(args)
    from msit_llm import DumpConfig, register_hook

    dump_config = DumpConfig(dump_path="./msit_dump_torch", dump_last_logits=True)
    register_hook(model, dump_config)

    if 'precision' in args.mode:
        precision(args, tokenizer, model)
    elif 'performance' in args.mode:
        performance(args, tokenizer, model)
    elif 'cli_demo' in args.mode:
        cli_demo(args, tokenizer, model)
    else:
        webUI(args, tokenizer, model)


if __name__ == '__main__':
    main()
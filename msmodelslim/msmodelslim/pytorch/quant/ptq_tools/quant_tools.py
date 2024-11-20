security.check_type(model_arch, str, param_name="model_arch")
        security.check_character(model_arch, param_name="model_arch")
        security.check_write_directory(save_path)
        security.check_element_type(input_names, str, list, param_name="input_names")
        security.check_character(input_names, param_name="input_names")
        check_type(fuse_add, bool, param_name="fuse_add")
        check_type(save_fp, bool, param_name="save_fp")

        self.export_onnx(model_arch, save_path, input_names)
        onnx_path = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
        onnx_path = get_valid_read_path(onnx_path)
        model = onnx.load(onnx_path)

        if "swinv2" in model_arch.lower() or "solov2" in model_arch.lower():
            from onnxsim import simplify
            model, check = simplify(model)

        graph = model.graph
        nodes = graph.node

        input_scale, input_offset, weight_scale, weight_offset, quant_weight = \
            self.get_quant_params()
        linear_params = ConvertLinearParams(
            onnx_model=model,
            input_scale=input_scale,
            input_offset=input_offset,
            weight_scale=weight_scale,
            weight_offset=weight_offset,
            quant_weight=quant_weight
        )
        input_scale, input_offset, weight_scale, weight_offset, quant_weight = \
            convert_linear_params(linear_params)
        quantized_weight_namd = []
        for item in nodes:
            if item.op_type == "Conv":
                weight_name = ".".join(item.input[1].split(".")[:-1])
                if weight_name in weight_scale.keys() and \
                        weight_scale.get(weight_name) is not None:
                    quantized_weight_namd.append(weight_name)
                    logger.info("Conv, item.name :%s, weight_name :%s ", item.name, weight_name)

            elif item.op_type == "MatMul":
                weight_name = item.input[1]
                if weight_name in weight_scale.keys() and \
                        weight_scale.get(weight_name) is not None:
                    quantized_weight_namd.append(weight_name)
                    logger.info("MatMul, item.name :%s, weight_name :%s ", item.name, weight_name)

        quantize_model_deploy_params = ModelDeployQuantParams(
            quantized_weight_name=quantized_weight_namd, 
            quant_weight_dict=quant_weight,
            input_scale_dict=input_scale,
            input_offset_dict=input_offset,
            weight_scale_dict=weight_scale,
            weight_offset_dict=weight_offset,
            fuse_add=fuse_add
        )
        quantize_model_deploy(graph, quantize_model_deploy_params)

        temp_quant_model_file = os.path.join(save_path, "{}_quant.onnx".format(model_arch))
        temp_quant_model_file = get_valid_write_path(temp_quant_model_file)
        with SafeWriteUmask():
            onnx.save(model, temp_quant_model_file)
            logger.info("Quantification ended and onnx is stored in %s ", temp_quant_model_file)

        if not save_fp:
            save_fp_path = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
            safe_delete_path_if_exists(save_fp_path)
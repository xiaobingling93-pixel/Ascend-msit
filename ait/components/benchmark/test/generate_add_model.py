import onnx
from onnx import helper
from onnx import TensorProto

# 定义输入维度
input_shape_1 = [-1, 3, -1, -1]
input_shape_2 = [-1, 3, -1, -1]

# 创建输入节点
input_1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape_1)
input_2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input_shape_2)

# 创建输出节点
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape_1)

# 创建输出节点
node_def = helper.make_node('Add', ['input1', 'input2'], ['output'], name='add1')

# 创建图
graph_def = helper.make_graph([node_def], 'test-add-model', [input_1, input_2], [output])

# 创建模型
model_def = helper.make_model(
    graph_def, producer_name='onnx-example', opset_imports=[helper.make_opsetid(domain="", version=11)]
)

# 保存模型
onnx.save(model_def, 'add_model.onnx')

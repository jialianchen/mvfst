#include <iostream>
#include <tensorflow/core/public/session.h>

#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"


/*
定义好图和保存点的路径
新建一个session
加载图结构
新建session
载入模型参数
构造输入数据

*/
using namespace std;
using namespace tensorflow;

int main()
{
    const string pathToGraph = "yourpath/demo_model/demo.meta";
    const string checkpointPath = "yourpath/demo_model/demo";
    auto session = NewSession(SessionOptions());
    if (session == nullptr)
    {
        throw runtime_error("Could not create Tensorflow session.");
        //cout << "Could not create Tensorflow session." << endl;
    }

    Status status;

// 读入我们预先定义好的模型的计算图的拓扑结构
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok())
    {
        // throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
        cout << "Error reading graph" << endl;
        return 0;
    }

// 利用读入的模型的图的拓扑结构构建一个session
    status = session->Create(graph_def.graph_def());
    if (!status.ok())
    {
        // throw runtime_error("Error creating graph: " + status.ToString());
        cout << "error creating graph" << endl;
        return 0;
    }

// 读入预先训练好的模型的权重
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok())
    {
        // throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
        cout << "error loading checkpoint" << endl;
        return 0;
    }

    // input相当于python版本中的feed_dict。
    std::vector<std::pair<string, Tensor>> input;
    // 输入Tensor的shape
    tensorflow::TensorShape inputshape({2, 2});
    // 根据类型和shape新建Tensor
    Tensor a(tensorflow::DT_INT32,inputshape);
    // 得到类型为int，维度为2的模板实例，类似于Eigen中矩阵的用法
    auto a_map = a.tensor<int,2>();
    int count = 1;
    for (int i=0; i<2; ++i){
        for (int j=0; j<2; ++j){
            a_map(i,j) = count++;

        }
    }
    // emplace_back用法类似于push_back,只是免去了构造结构体或类的麻烦
    input.emplace_back(std::string("x"), a);

//   运行模型，并获取输出
    std::vector<tensorflow::Tensor> answer;
    status = session->Run(input, {"res"}, {}, &answer);

    Tensor result = answer[0];
    auto result_map = result.tensor<int,2>();
    cout<<"result: "<<result_map(0,0)<<endl;
    cout<<"result: "<<result_map(1,0)<<endl;

    return 0;
}
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

/*


原代码来自NewReno.cpp
目前没有对要存储的数据进行处理，直接保存了
可以简单扩展到其他传统方法

但，我不知道怎么测试。。。大概没问题。。。起码没语法问题。。。


然后更新了c调用tf的内容，

需要模型保存的地址


*/
#include <glog/logging.h>
#include <quic/congestion_control/CongestionControlFunctions.h>
#include <quic/congestion_control/NN.h>
#include <quic/logging/QLoggerConstants.h>
#include "hdf5.h" //引入h5的处理文件
#include "utils/TFUtils.hpp" //这一步引入了调用c语言的头文件
//#define duration 10000 //定义这个线程的持续时间，感觉基本没有用

unsigned int step_len_ms = 5; // 5s，定义更新时间
namespace quic {
constexpr int kRenoLossReductionFactorShift = 1;
NN::NN(QuicConnectionStateBase& conn)
    : conn_(conn),
      ssthresh_(std::numeric_limits<uint32_t>::max()),
      delivered(0),
      lastdelivered(0),
      lastsent_bytes(0),
      delivery_rate_ewma(0),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen) {
  cwndBytes_ = boundedCwnd(
      cwndBytes_,
      conn_.udpSendPacketLen,
      conn_.transportSettings.maxCwndInMss,
      conn_.transportSettings.minCwndInMss);
  memset(infor.a, 0, sizeof(infor.a));
  inforlist.clear();
  VLOG(google::INFO) << "NN start";
  lasttime = Clock::now();

  // std::thread t1(&NN::TimerThread, this); //新建监督线程
}

void NN ::fresh() {
  //实际上接下来没注意量纲，得到的数据没有进行处理
  // newinfor.cwnd = getCongestionWindow();
  // newinfor.snd_ssthresh = ssthresh_;
  // newinfor.time_delta =
  //     (unsigned int)((Clock::now() - lasttime).count() / 1000000);
  // lasttime = Clock::now();
  // newinfor.srtt_ms = conn_.lossState.srtt.count();
  // newinfor.pacing_rate = conn_.lossState.totalBytesSent /
  // newinfor.time_delta; newinfor.lost_rate = newinfor.lost_bytes /
  // newinfor.time_delta; newinfor.packets_out = getBytesInFlight();
  // newinfor.retrans_out = conn_.lossState.totalBytesRetransmitted;
  // // newinfo.mss = ; //udp好像没有mss
  // inforlist.push_back(newinfor);

  //接下来，把信息存到h5，用作训练
  // vector没问题，每次增加一个
  // 最后的s格式是，300*5*8，维护一个vector，里面是一个二维数组，5种数据，长度为8，滚动更新如何做到？
  //先O(8)的更新吧
  // 300*1的
  // 300*7的概率
  // 300*7的动作
  // 5种数据
  //

  /*
   每次收到ack进行更新
   判断是否到了更新的时间.
   然后就是找到每个值
   rtt = float(curr_time_ms - ack.send_ts)
   delay = rtt - self.min_rtt
   if self.delay_ewma is None:
            self.delay_ewma = delay
        else:
            self.delay_ewma = 0.875 * self.delay_ewma + 0.125 * delay

  self.delivered += ack.ack_bytes
  self.delivered_time = curr_time_ms
  delivery_rate = (0.008 * (self.delivered - ack.delivered) / max(1,
self.delivered_time - ack.delivered_time)) 即totalBytesSentThen; if
self.delivery_rate_ewma is None: self.delivery_rate_ewma = delivery_rate else:
            self.delivery_rate_ewma = (
                0.875 * self.delivery_rate_ewma + 0.125 * delivery_rate)
 send_rate = 0.008 * (self.sent_bytes - ack.sent_bytes) / max(1, rtt)
 LossState.totalBytesSent
 用这个去保存
   s = [self.delay_ewma / 100.,
                     self.delivery_rate_ewma / 10.,
                     self.send_rate_ewma / 10.,
                     self.cwnd / 200.,
                     duration_ / 100.]


  */
  VLOG(google::INFO) << "fresh";
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 7; j++) {
      infor.a[i][j] = infor.a[i][j + 1];
    }
  }

  infor.a[0][7] = conn_.lossState.srtt.count() / 100;
  infor.a[1][7] = delivery_rate_ewma / 10;
  //std::cout<<delivery_rate_ewma<<std::endl;
  infor.a[2][7] = send_rate_ewma / 10;
  infor.a[3][7] = getCongestionWindow() / 200;
  uint64_t duration =
      (unsigned int)((Clock::now() - lasttime).count() / 1000000);
  infor.a[4][7] = duration;

  inforlist.push_back(infor);

  nodedim.d0 = inforlist.size();
  nodedim.d1 = 5;
  nodedim.d2 = 8;
  hid_t file_id;
  herr_t status;
  file_id = H5Fcreate("my_file_test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  float p[inforlist.size()][5][8];
  //   int*** p = new int**[inforlist.size()]; //开辟行
  //   for (int i = 0; i < inforlist.size(); i++) {
  //     *p[i] = new int[5]; //开辟列
  //     for (int j = 0; j < 5; j++) {
  //       p[i][j] = new int[8];
  //     }
  //   }
  for (int i = 0; i < inforlist.size(); i++) {
    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 8; k++) {
        p[i][j][k] = inforlist[i].a[i][j];
      }
    }
  }
 
  saveh5(file_id, status, p, 3, nodedim, "/s");


  //写完了s的h5文件，继续写r,p,d,a的文件
  //计算r
  /*
  所以，每一次收到ack后更新一部分数据
  然后到一定的更新时间后更新总体数据
   cum_bytes = np.sum(self.curr_bytes) 就是数据量总和，（多个ack的问题
        # bytes / 1000 / ms -> kb / ms -> mb / s
        _throughput = 0.008 * cum_bytes / (duration + 1e-6)
        _delay = np.mean(self.curr_delay)（保存多个
        _delay_max = 1000.
        _throughput_max = 100.
        # ocra loss
        _reward = (_throughput / (_delay + 1e-6)) / (_throughput_max /
  _delay_max)
  */
  uint64_t cum_bytes = 0;
  for (int i = 0; i < curr_bytes.size(); i++) {
    cum_bytes += curr_bytes[i];
  }
  uint64_t _throughput = 0.008 * cum_bytes / (duration + 1e-6);
  float sum =
      std::accumulate(std::begin(curr_delay), std::end(curr_delay), 0.0);
  float _delay = sum / curr_delay.size();
  uint64_t _delay_max = 1000;
  uint64_t _throughput_max = 10;
  float _reward =
      1.0 * (_throughput / (_delay + 1e-6)) / (_throughput_max / _delay_max);
  r.push_back(_reward);
  nodedim.d0 = r.size();
  float r_add[r.size()];

  for (int i = 0; i < r.size(); i++) {
    r_add[i] = r[i];
  }
  saveh5(file_id, status, r_add, 1, nodedim, "/r");

  /*
  计算a和p，同理
  */

  // alpha= coreNN->getInstance()->Predict(infolist);
  // new_cwnd =cwndBytes_*alpha;

  /*
  下面的逻辑，每次新开一次session，然后读入模型文件，输入参数，预测读数，关闭session，注意需要模型路径
  */

  // Load graph

  struct node tmpnode;
  std::string graph_path =
      "/home/godka/jlc/dugu/monitor/model/test.pb"; // argv[1];
  // TFUtils init
  TFUtils TFU;
  TFUtils::STATUS statustf = TFU.LoadModel(graph_path);

  if (statustf != TFUtils::SUCCESS) {
    std::cerr << "Can't load graph" << std::endl;
  }
  const std::vector<std::int64_t> input_infor_dims = {1, 5, 8};

 

  std::vector<node> input_infor_vals;
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<8;j++)
        {
            std::cout<<infor.a[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
  input_infor_vals.push_back(infor);
  //    VLOG(google::INFO) << input_infor_vals.size();
  //     VLOG(google::INFO) << "test4.1";
  const std::vector<TF_Output> input_ops = {TFU.GetOperationByName("state", 0)};
  // const std::vector<TF_Tensor*> input_tensors = {
  //   TFUtils::CreateTensor(TF_FLOAT, input_infor_dims.data(),
  //   input_infor_dims.data().size(),input_infor_vals.data(),)};
  const std::vector<TF_Tensor*> input_tensors = {
      TFUtils::CreateTensor(TF_FLOAT, input_infor_dims, input_infor_vals)};

  const std::vector<TF_Output> output_ops = {
      TFU.GetOperationByName("output", 0)};

  std::vector<TF_Tensor*> output_tensors = {nullptr};


  statustf =
      TFU.RunSession(input_ops, input_tensors, output_ops, output_tensors);


  TFUtils::PrinStatus(statustf);

  const std::vector<std::vector<float>> data =
      TFUtils::GetTensorsData<float>(output_tensors);

  const std::vector<float> action_prob = data[0];
  // const std::vector<std::vector<float>> data =
  // TFUtils::GetTensorsData<float>(output_tensors); const std::vector<float>
  // result = data[0];
  std::cout << "Output value: " << action_prob[0] << std::endl;

  TFUtils::DeleteTensors(input_tensors);
  TFUtils::DeleteTensors(output_tensors);

  //   std::string graph_path = "/home/godka/jlc/dugu/monitor/model/test.pb"; //
  //   argv[1];//这一步需要定义模型保存的路径

  //   // TFUtils init
  //   TFUtils TFU;
  //   TFUtils::STATUS statustf = TFU.LoadModel(graph_path);

  //   if (statustf != TFUtils::SUCCESS) {
  //     std::cerr << "Can't load graph" << std::endl;
  //     return;
  //   }

  //   // Load image and convert to tensor
  //   // int batch_size = inforlist.size();
  //   const std::vector<std::int64_t> input_infor_dims = {1,5, 8};
  //   //float input[5][8];
  // //   std::vector<std::vector<float> > input(5, std::vector<float>(8, 0));
  // //   for(int i=0;i<5;i++)
  // //   {
  // //       for(int j=0;j<8;j++)
  // //       {
  // //         input[i][j]=infor.a[i][j];
  // //       }
  // //   }
  //   std::vector<node>input_infor_vals;
  //   //std::vector<std::vector<std::vector<float> > > input_infor_vals;
  //   input_infor_vals.push_back(infor);
  //    VLOG(google::INFO) << input_infor_vals.size();
  //     VLOG(google::INFO) << "test4.1";
  //   const std::vector<TF_Output> input_ops = {TFU.GetOperationByName("state",
  //   0)};
  //     // const std::vector<TF_Tensor*> input_tensors = {
  //     //   TFUtils::CreateTensor(TF_FLOAT, input_infor_dims.data(),
  //     input_infor_dims.data().size(),input_infor_vals.data(),)};
  //   const std::vector<TF_Tensor*> input_tensors = {
  //       TFUtils::CreateTensor(TF_FLOAT, input_infor_dims, input_infor_vals)};
  //  VLOG(google::INFO) <<input_tensors.size();
  //   // Output Tensor/Ops Create
  //   const std::vector<TF_Output> output_ops = {
  //       TFU.GetOperationByName("jia_ouput", 0)};

  //   std::vector<TF_Tensor*> output_tensors = {nullptr};
  //     VLOG(google::INFO) << "test4.2";
  //   statustf =
  //       TFU.RunSession(input_ops, input_tensors, output_ops, output_tensors);
  //     VLOG(google::INFO) << "test4.3";
  //   const std::vector<std::vector<float>> data =
  //       TFUtils::GetTensorsData<float>(output_tensors);
  //       VLOG(google::INFO) << "test4.4";
  //   const std::vector<float> action_prob = data[0];

  float tmpmin = 9;
  int act = 0;
  for (int i = 0; i < action_prob.size(); i++) {
    if (tmpmin > action_prob[i]) {
      tmpmin = action_prob[i];
      act = i;
    }
  }

  node2 action_vec;
  memset(action_vec.a, 0, sizeof(action_vec.a));
  // action_vec = np.zeros(A_DIM)
  action_vec.a[act] = 1;
  //std::cout <<"act="<<" "<<act<<std::endl;
  a_batch.push_back(action_vec);
  node2 newaction_prob;
  for (int i = 0; i < 7; i++)

  {
    newaction_prob.a[i] = action_prob[i];
  }
  p_batch.push_back(newaction_prob);
  nodedim.d0 = a_batch.size();
  nodedim.d1 = 7;
  float p2[a_batch.size()][7];
  //   int** p2 = new int*[7]; //开辟行
  //   for (int i = 0; i < 7; i++)
  //     p2[i] = new int[a_batch.size()]; //开辟列
  for (int i = 0; i < a_batch.size(); i++) {
    for (int j = 0; j < 7; j++) {
      p2[i][j] = a_batch[i].a[j];
    }
  }
  saveh5(file_id, status, p2, 2, nodedim, "/a");
  for (int i = 0; i < a_batch.size(); i++) {
    for (int j = 0; j < 7; j++) {
      p2[i][j] = p_batch[i].a[j];
    }
  }
  saveh5(file_id, status, p2, 2, nodedim, "/p");

  status = H5Fclose(file_id);
  

  uint64_t new_cwnd = cwndBytes_ * ACTIONS[act] ; //正常是上面的调用
  // newinfor.init();
  cwndBytes_ = boundedCwnd(
      new_cwnd,
      conn_.udpSendPacketLen,
      conn_.transportSettings.maxCwndInMss,
      conn_.transportSettings.minCwndInMss);
 
}
void NN::saveh5(
    hid_t file_id,
    herr_t status,
    void* p,
    int rank,
    dim d,
    char* s) {
  hid_t dataspace_id;
  if (rank == 3) {
    hsize_t dims[3];
    dims[0] = d.d0;
    dims[1] = d.d1;
    dims[2] = d.d2;
    dataspace_id = H5Screate_simple(rank, dims, NULL);
  }
  if (rank == 2) {
    hsize_t dims[2];
    dims[0] = d.d0;
    dims[1] = d.d1;

    dataspace_id = H5Screate_simple(rank, dims, NULL);
  }
  if (rank == 1) {
    hsize_t dims[1];
    dims[0] = d.d0;

    dataspace_id = H5Screate_simple(rank, dims, NULL);
  }
  hid_t dataset_id; // 数据集本身的id
  dataset_id = H5Dcreate(
      file_id,
      s,
      H5T_NATIVE_FLOAT,
      dataspace_id,
      H5P_DEFAULT,
      H5P_DEFAULT,
      H5P_DEFAULT);

  status =
      H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, p);
  status = H5Dclose(dataset_id);
  status = H5Sclose(dataspace_id);
}
void NN ::saveack(const AckEvent& ack) {
  VLOG(google::INFO) << "saveack";
  uint64_t duration_ = 0;
  // std::time_t tt_last;
  // std::time_t tt_now;
  // tt_last = std::chrono::system_clock::to_time_t (lasttime);
  // tt_now=std::chrono::system_clock::to_time_t (Clock::now());

  duration_ = (float)((Clock::now() - lasttime).count() / 1000000);

//  VLOG(google::INFO) << duration_; //<< " " <<  tt_last  << " "
                                   //<<tt_now;

  for (const auto& packet : ack.ackedPackets) {
    delivered += packet.encodedSize;
    curr_bytes.push_back(packet.encodedSize);
    float rtt = float((Clock::now() - packet.sentTime).count());
    min_rtt = std::min(min_rtt, rtt);
    float delay = rtt - min_rtt;
    curr_delay.push_back(delay);
    float send_rate = 0.008 *
        (conn_.lossState.totalBytesSent - lastsent_bytes) /
        std::max((float)1, rtt);
    if (send_rate_ewma  == 0) {
      send_rate_ewma = send_rate;

    } else {
      send_rate_ewma = 0.875 * send_rate_ewma + 0.125 * send_rate;
    }
    lastsent_bytes = conn_.lossState.totalBytesSent;
  }
  
  float delivery_rate = 0.008 * (delivered - lastdelivered) /
      std::max(1,((int)((Clock::now() - lasttime).count() / 1000000)));
 //   std::cout<<"delivered="<<delivered<<" lastdelivered="<<lastdelivered<<" time="<<((int)((Clock::now() - lasttime).count() / 1000000))<<std::endl;
  //  ((unsigned int)（Clock::now() - lasttime).count));
  // LOG(INFO) <<
  if (delivery_rate_ewma == 0) {
    delivery_rate_ewma = delivery_rate;

  } else

  {
    //  std::cout<<delivery_rate_ewma<<" "<<delivery_rate<<std::endl;
    delivery_rate_ewma = (0.875 * delivery_rate_ewma + 0.125 * delivery_rate);
  }
  lastdelivered = delivered;

  // (unsigned int)((Clock::now() - lasttime).count() / 1000000);
  if (duration_ > step_len_ms) {
    fresh();
    lasttime = Clock::now();
  }
}

/*void NN ::saveloss(const LossEvent& loss) {
  newinfor.lost_bytes += loss.lostBytes;
}*/

//往下都是原NewReno.cpp代码，只在onAckEvent，onPacketLoss两处调用了保存函数
/*
更改为dugu方法，实际上就是把原本的控制方法全部去除，纯强化学习方法。
先简单修改完，然后进行下一步搭建服务器。
*/
void NN::onRemoveBytesFromInflight(uint64_t bytes) {
  subtractAndCheckUnderflow(conn_.lossState.inflightBytes, bytes);
  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_
           << " inflight=" << conn_.lossState.inflightBytes << " " << conn_;
  if (conn_.qLogger) {
    conn_.qLogger->addCongestionMetricUpdate(
        conn_.lossState.inflightBytes, getCongestionWindow(), kRemoveInflight);
  }
}

void NN::onPacketSent(const OutstandingPacket& packet) {
  addAndCheckOverflow(
      conn_.lossState.inflightBytes, packet.metadata.encodedSize);

  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_
           << " inflight=" << conn_.lossState.inflightBytes
           << " packetNum=" << packet.packet.header.getPacketSequenceNum()
           << " " << conn_;
  if (conn_.qLogger) {
    conn_.qLogger->addCongestionMetricUpdate(
        conn_.lossState.inflightBytes,
        getCongestionWindow(),
        kCongestionPacketSent);
  }
}

void NN::onAckEvent(const AckEvent& ack) {
  DCHECK(ack.largestAckedPacket.has_value() && !ack.ackedPackets.empty());
  subtractAndCheckUnderflow(conn_.lossState.inflightBytes, ack.ackedBytes);
  saveack(ack);
  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_
           << " inflight=" << conn_.lossState.inflightBytes << " " << conn_;
  if (conn_.qLogger) {
    conn_.qLogger->addCongestionMetricUpdate(
        conn_.lossState.inflightBytes,
        getCongestionWindow(),
        kCongestionPacketAck);
  }
  for (const auto& packet : ack.ackedPackets) {
    onPacketAcked(packet);
  }
  cwndBytes_ = boundedCwnd(
      cwndBytes_,
      conn_.udpSendPacketLen,
      conn_.transportSettings.maxCwndInMss,
      conn_.transportSettings.minCwndInMss);
}
void NN::onPacketAcked(
    const CongestionController::AckEvent::AckPacket& packet) {
  if (endOfRecovery_ && packet.sentTime < *endOfRecovery_) {
    return;
  }

  if (cwndBytes_ < ssthresh_) {
    addAndCheckOverflow(cwndBytes_, packet.encodedSize);
  } else {
    // TODO: I think this may be a bug in the specs. We should use
    // conn_.udpSendPacketLen for the cwnd calculation. But I need to
    // check how Linux handles this.
    uint64_t additionFactor =
        (kDefaultUDPSendPacketLen * packet.encodedSize) / cwndBytes_;
    addAndCheckOverflow(cwndBytes_, additionFactor);
  }
}

void NN::onPacketAckOrLoss(
    folly::Optional<AckEvent> ackEvent,
    folly::Optional<LossEvent> lossEvent) {
  if (lossEvent) {
    //onPacketLoss(*lossEvent);
    // When we start to support pacing in NewReno, we need to call onPacketsLoss
    // on the pacer when there is loss.
  }
  if (ackEvent && ackEvent->largestAckedPacket.has_value()) {
    onAckEvent(*ackEvent);
  }
  // TODO: Pacing isn't supported with NewReno
}

/* void NN::onPacketLoss(const LossEvent& loss) {
  DCHECK(
      loss.largestLostPacketNum.has_value() &&
      loss.largestLostSentTime.has_value());
  // saveloss(loss);
  subtractAndCheckUnderflow(conn_.lossState.inflightBytes, loss.lostBytes);
  if (!endOfRecovery_ || *endOfRecovery_ < *loss.largestLostSentTime) {//主要操作在这里。
//拥塞窗口减半，权限等于拥塞窗口
    endOfRecovery_ = Clock::now();
    cwndBytes_ = (cwndBytes_ >> kRenoLossReductionFactorShift);
    cwndBytes_ = boundedCwnd(
        cwndBytes_,
        conn_.udpSendPacketLen,
        conn_.transportSettings.maxCwndInMss,
        conn_.transportSettings.minCwndInMss);
    // This causes us to exit slow start.
    ssthresh_ = cwndBytes_;
    VLOG(10) << __func__ << " exit slow start, ssthresh=" << ssthresh_
             << " packetNum=" << *loss.largestLostPacketNum
             << " writable=" << getWritableBytes() << " cwnd=" << cwndBytes_
             << " inflight=" << conn_.lossState.inflightBytes << " " << conn_;
  } else {
    VLOG(10) << __func__ << " packetNum=" << *loss.largestLostPacketNum
             << " writable=" << getWritableBytes() << " cwnd=" << cwndBytes_
             << " inflight=" << conn_.lossState.inflightBytes << " " << conn_;
  }

  if (conn_.qLogger) {
    conn_.qLogger->addCongestionMetricUpdate(
        conn_.lossState.inflightBytes,
        getCongestionWindow(),
        kCongestionPacketLoss);
  }
  if (loss.persistentCongestion) {`   
    VLOG(10) << __func__ << " writable=" << getWritableBytes()
             << " cwnd=" << cwndBytes_
             << " inflight=" << conn_.lossState.inflightBytes << " " << conn_;
    if (conn_.qLogger) {
      conn_.qLogger->addCongestionMetricUpdate(
          conn_.lossState.inflightBytes,
          getCongestionWindow(),
          kPersistentCongestion);
    }
    cwndBytes_ = conn_.transportSettings.minCwndInMss * conn_.udpSendPacketLen;
  }
}
 */
uint64_t NN::getWritableBytes() const noexcept {
  if (conn_.lossState.inflightBytes > cwndBytes_) {
    return 0;
  } else {
    return cwndBytes_ - conn_.lossState.inflightBytes;
  }
}

uint64_t NN::getCongestionWindow() const noexcept {
  return cwndBytes_;
}

bool NN::inSlowStart() const noexcept {
  return cwndBytes_ < ssthresh_;
}

CongestionControlType NN::type() const noexcept {
  return CongestionControlType::NewReno;
}

uint64_t NN::getBytesInFlight() const noexcept {
  return conn_.lossState.inflightBytes;
}

void NN::setAppIdle(bool, TimePoint) noexcept { /* unsupported */
}

void NN::setAppLimited() { /* unsupported */
}

bool NN::isAppLimited() const noexcept {
  return false; // unsupported
}

} // namespace quic

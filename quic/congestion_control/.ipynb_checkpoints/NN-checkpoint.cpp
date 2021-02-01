/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

/*
主要思路是
定义一个监督线程
定时存储和调用强化学习接口进行更新拥塞窗口

原代码来自NewReno.cpp
目前没有对要存储的数据进行处理，直接保存了
可以简单扩展到其他传统方法

但，我不知道怎么测试。。。大概没问题。。。起码没语法问题。。。


然后更新了c调用tf的内容，

需要模型保存的地址


*/

#include <quic/congestion_control/CongestionControlFunctions.h>
#include <quic/congestion_control/Nn.h>
#include <quic/logging/QLoggerConstants.h>
#include "utils/TFUtils.hpp"//这一步引入了调用c语言的头文件
#define duration 10000//定义这个线程的持续时间，感觉基本没有用


unsigned int report_period = 5; // 5s，定义更新时间
namespace quic {
constexpr int kRenoLossReductionFactorShift = 1;
NN::NN(QuicConnectionStateBase& conn)
    : conn_(conn),
      ssthresh_(std::numeric_limits<uint32_t>::max()),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen) {
  cwndBytes_ = boundedCwnd(
      cwndBytes_,
      conn_.udpSendPacketLen,
      conn_.transportSettings.maxCwndInMss,
      conn_.transportSettings.minCwndInMss);
  newinfor.init();
  std::thread t1(&NN::TimerThread, this);//新建监督线程
}

void NN::TimerThread() //当做监督线程，定时存储
{
  starttime = Clock::now(); // starttime 从第一次开始计时
  lasttime = starttime;

  bool send_traffic = true;
  if (duration !=0) //从orca源码上面参考下来的，实际上感觉这里没啥用
  {
    while (send_traffic) {
      usleep(report_period * 1000);//orca定义的存储周期是5s
      auto elapsed = ((Clock::now() - starttime) / 1000000);//orca用来控制线程运行时间的，这里没有用
      if (elapsed.count() > duration) {
        send_traffic = false;
      }
      saveandfresh();//存储并且更新
    }
  }

  return;
}

void NN ::saveandfresh() {
  //实际上接下来没注意量纲，得到的数据没有进行处理
  newinfor.cwnd = getCongestionWindow();
  newinfor.snd_ssthresh = ssthresh_;
  newinfor.time_delta =
      (unsigned int)((Clock::now() - lasttime).count() / 1000000);
  lasttime = Clock::now();
  newinfor.srtt_ms = conn_.lossState.srtt.count();
  newinfor.pacing_rate = conn_.lossState.totalBytesSent / newinfor.time_delta;
  newinfor.lost_rate = newinfor.lost_bytes / newinfor.time_delta;
  newinfor.packets_out = getBytesInFlight();
  newinfor.retrans_out = conn_.lossState.totalBytesRetransmitted;
  // newinfo.mss = ; //udp好像没有mss
  inforlist.push_back(newinfor);





  // alpha= coreNN->getInstance()->Predict(infolist);
  // new_cwnd =cwndBytes_*alpha;

  /*
  下面的逻辑，每次新开一次session，然后读入模型文件，输入参数，预测读数，关闭session，注意需要模型路径
  */

 
    // Load graph
    std::string graph_path = "tmp";//argv[1];//这一步需要定义模型保存的路径
 
    // TFUtils init
    TFUtils TFU;
    TFUtils::STATUS status = TFU.LoadModel(graph_path);
 
    if (status != TFUtils::SUCCESS) {
        std::cerr << "Can't load graph" << std::endl;
        return ;
    }
 
    // Load image and convert to tensor
    int batch_size=inforlist.size();
    const std::vector<std::int64_t> input_infor_dims = {batch_size, 8};
     std::vector<infor> input_infor_vals = inforlist;
    
  
 
    const std::vector<TF_Output> input_ops = {TFU.GetOperationByName("infor", 0)};
 
    const std::vector<TF_Tensor*> input_tensors = {TFUtils::CreateTensor(TF_FLOAT, input_infor_dims, input_infor_vals)};
    // Output Tensor/Ops Create
    const std::vector<TF_Output> output_ops = {TFU.GetOperationByName("result", 0)};
 
    std::vector<TF_Tensor*> output_tensors = {nullptr};
 
    status = TFU.RunSession(input_ops, input_tensors,
                            output_ops, output_tensors);
    
    
    const std::vector<std::vector<float>> data = TFUtils::GetTensorsData<float>(output_tensors);
    const std::vector<float> result = data[0];
 
  
    TFUtils::DeleteTensors(input_tensors);
    TFUtils::DeleteTensors(output_tensors);




  uint64_t new_cwnd = cwndBytes_ *result[0];//正常是上面的调用
  newinfor.init();
  cwndBytes_ = boundedCwnd(
      new_cwnd,
      conn_.udpSendPacketLen,
      conn_.transportSettings.maxCwndInMss,
      conn_.transportSettings.minCwndInMss);
}


void NN ::saveack(const AckEvent& ack) {
  
  for (const auto& packet : ack.ackedPackets) {
    if (newinfor.min_rtt != 0)
    newinfor.min_rtt =std::min(newinfor.min_rtt,(uint64_t)(ack.ackTime - packet.sentTime).count());
  else
    newinfor.min_rtt =(uint64_t)(ack.ackTime - packet.sentTime).count();
    newinfor.thr += packet.encodedSize;
    newinfor.cnt = newinfor.cnt + 1;
  }

  
  if (newinfor.max_packets_out != 0)
    newinfor.max_packets_out =std::max(conn_.lossState.inflightBytes, newinfor.max_packets_out);
  else
    newinfor.max_packets_out = conn_.lossState.inflightBytes;
}

void NN ::saveloss(const LossEvent& loss) {
  newinfor.lost_bytes += loss.lostBytes;
}


//往下都是原NewReno.cpp代码，只在onAckEvent，onPacketLoss两处调用了保存函数
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
    onPacketLoss(*lossEvent);
    // When we start to support pacing in NewReno, we need to call onPacketsLoss
    // on the pacer when there is loss.
  }
  if (ackEvent && ackEvent->largestAckedPacket.has_value()) {
    onAckEvent(*ackEvent);
  }
  // TODO: Pacing isn't supported with NewReno
}

void NN::onPacketLoss(const LossEvent& loss) {
  DCHECK(
      loss.largestLostPacketNum.has_value() &&
      loss.largestLostSentTime.has_value());
  saveloss(loss);
  subtractAndCheckUnderflow(conn_.lossState.inflightBytes, loss.lostBytes);
  if (!endOfRecovery_ || *endOfRecovery_ < *loss.largestLostSentTime) {
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
  if (loss.persistentCongestion) {
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

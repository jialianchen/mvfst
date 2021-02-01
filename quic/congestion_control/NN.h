/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#pragma once

#include <quic/QuicException.h>
#include <quic/state/StateData.h>
#include <sys/select.h>
#include <limits>
#include <chrono>
#include <sys/shm.h>
#include <time.h>
#include "hdf5.h" //引入h5的处理文件
namespace quic {

class NN : public CongestionController {
 public:
  explicit NN(QuicConnectionStateBase& conn);

  //下面是新定义的部分
  std::chrono::_V2::steady_clock::time_point starttime;
  std::chrono::_V2::steady_clock::time_point lasttime;
  float alpha;
  void TimerThread();

  void fresh();
  void saveloss(const LossEvent& loss);
  void saveack(const AckEvent&);
  struct dim {
  hsize_t d0, d1, d2;
  } nodedim;
  void saveh5(hid_t file_id, herr_t status, void* p, int rank, dim d, char*);
  struct node
  {
    float a[5][8];
  }infor;
  std::vector<node>inforlist;
  std::vector<float >r;
  struct node2
  {
    float a[7];
  };
  std::vector<node2>a_batch;
  std::vector<node2>p_batch;
  std::vector<float >curr_bytes;
  std::vector<float >curr_delay;
  float min_rtt;
  float  delivery_rate_ewma;
  float  lastdelivered;
  float  lastsent_bytes;
  float  delivered;
  float  send_rate_ewma;
  
  //下面是之前的NEWReno.h的内容
  void onRemoveBytesFromInflight(uint64_t) override;
  void onPacketSent(const OutstandingPacket& packet) override;
  void onPacketAckOrLoss(folly::Optional<AckEvent>, folly::Optional<LossEvent>)
      override;

  uint64_t getWritableBytes() const noexcept override;
  uint64_t getCongestionWindow() const noexcept override;
  void setAppIdle(bool, TimePoint) noexcept override;
  void setAppLimited() override;

  CongestionControlType type() const noexcept override;

  bool inSlowStart() const noexcept;

  uint64_t getBytesInFlight() const noexcept;

  bool isAppLimited() const noexcept override;
  float ACTIONS[7] = {-0.5,-0.3, -0.1, 0, 0.1, 0.2, 0.4};
  int A_DIM = 7;
  


 private:
  void onPacketLoss(const LossEvent&);
  void onAckEvent(const AckEvent&);
  void onPacketAcked(const CongestionController::AckEvent::AckPacket&);

 private:
  QuicConnectionStateBase& conn_;
  uint64_t ssthresh_;
  uint64_t cwndBytes_;
  
  
  folly::Optional<TimePoint> endOfRecovery_;

  
};
} // namespace quic

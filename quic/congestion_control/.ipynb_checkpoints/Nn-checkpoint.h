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

#include <sys/shm.h>
#include <time.h>
namespace quic {

class NN : public CongestionController {
 public:
  explicit NN(QuicConnectionStateBase& conn);

  //下面是新定义的部分
  std::chrono::_V2::steady_clock::time_point starttime;
  std::chrono::_V2::steady_clock::time_point lasttime;
  double alpha;
  void TimerThread();

  void saveandfresh();
  void saveloss(const LossEvent& loss);
  void saveack(const AckEvent&);

  struct infor {
    uint64_t min_rtt; /* min-filtered RTT in uSec */
    uint64_t avg_urtt; /* averaged RTT in uSec from the previous info request
                          till now*/
    uint64_t cnt; /* number of RTT samples uSed for averaging */
    unsigned long thr; /*Bytes per second*/
    uint64_t thr_cnt;
    uint64_t cwnd;
    uint64_t pacing_rate;
    uint64_t lost_bytes;
    uint64_t srtt_us; /* smoothed round trip time << 3 in usecs */
    uint64_t snd_ssthresh; /* Slow start size threshold*/
    uint64_t packets_out; /* Packets which are "in flight"*/
    uint64_t retrans_out; /* Retransmitted packets out*/
    uint64_t max_packets_out; /* max packets_out in last window */
    //uint64_t mss;  udp好像没有mss
    uint64_t time_delta;
    uint64_t srtt_ms;
    uint64_t lost_rate;

    void init() {
      min_rtt = 0;
      avg_urtt = 0;
      cnt = 0;
      thr = 0;
      thr_cnt = 0;
      cwnd = 0;
      pacing_rate = 0;
      lost_bytes = 0;
      srtt_us = 0;
      snd_ssthresh = 0;
      retrans_out = 0;
      max_packets_out = 0;
      //mss = 0;
      time_delta = 0;
      srtt_ms = 0;
      lost_rate = 0;
    }
  };
  
  infor newinfor;
  std::vector<infor> inforlist;


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

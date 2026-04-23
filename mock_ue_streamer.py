import socket
import json
import time
import random

def run_mock_streamer(host="127.0.0.1", port=9000, interval=0.5):
    """
    模拟 workflow_infer.py 发送给 Unreal Engine (ADasVisualizerActor) 的数据包
    """
    # 创建 UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"Starting mock streamer to {host}:{port}...")
    print("Press Ctrl+C to stop.")

    # 模拟一些行人 (ID 和 初始位置)
    pedestrians = [
        {"id": 1, "pos": 100.0, "speed": 3.0},
        {"id": 2, "pos": 250.0, "speed": -5.0}
    ]

    try:
        while True:
            # 1. 模拟发送 'signal' 包 (会被 C++ 过滤，但 Python 脚本实际上会发)
            # signal_packet = {
            #     "packet_type": "signal",
            #     "timestamp": time.time(),
            #     "data": [random.uniform(-1, 1) for _ in range(10)]
            # }
            # sock.sendto((json.dumps(signal_packet) + "\n").encode("utf-8"), (host, port))

            # 2. 模拟发送 'event' 包 (C++ 脚本主要接收这个)
            for p in pedestrians:
                # 随机更新位置模拟走动
                p["pos"] += p["speed"] + random.uniform(-0.2, 0.2)
                
                # 构造符合 C++ TryGetNumberField 的 payload
                event_packet = {
                    "packet_type": "event",
                    "timestamp": time.time(),
                    "channel_index": round(p["pos"]),  # 模拟通道索引
                    "confidence": random.uniform(0.7, 0.98)
                }

                # 发送数据，加上 \n 确保 C++ ParseIntoArray 正常工作
                message = (json.dumps(event_packet) + "\n").encode("utf-8")
                sock.sendto(message, (host, port))
                
                print(f"Sent event: Channel={event_packet['channel_index']}, Conf={event_packet['confidence']:.2f}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStreamer stopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    # 默认端口 9000 是常见的测试端口，建议根据 UE 里的 UdpListenPort 修改
    run_mock_streamer(port=9000, interval=0.5)

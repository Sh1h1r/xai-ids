import time
from collections import defaultdict

FLOW_TIMEOUT = 3

flows = defaultdict(lambda: {
    "start_time": None,
    "packet_count": 0,
    "total_bytes": 0,
    "packet_lengths": []
})


def get_flow_key(packet):
    try:
        if packet.haslayer("IP"):
            ip = packet["IP"]
            proto = ip.proto

            if packet.haslayer("TCP") or packet.haslayer("UDP"):
                sport = packet.sport
                dport = packet.dport
            else:
                sport = 0
                dport = 0

            return (ip.src, ip.dst, sport, dport, proto)

    except Exception:
        return None

    return None


def update_flow(packet):
    key = get_flow_key(packet)

    if key is None:
        return None

    now = time.time()
    flow = flows[key]

    if flow["start_time"] is None:
        flow["start_time"] = now

    pkt_len = len(packet)

    flow["packet_count"] += 1
    flow["total_bytes"] += pkt_len
    flow["packet_lengths"].append(pkt_len)

    if now - flow["start_time"] >= FLOW_TIMEOUT:
        completed_flow = flows.pop(key)
        return build_features(completed_flow, key)

    return None


def build_features(flow, key):
    src, dst, sport, dport, proto = key

    packet_lengths = flow["packet_lengths"]

    return {
        "Destination Port": dport,
        "Packet Length Mean": sum(packet_lengths) / len(packet_lengths),
        "Fwd Packet Length Max": max(packet_lengths),
        "Total Length of Fwd Packets": flow["total_bytes"],

        # used only for live rule, not ML model
        "Flow Packet Count": flow["packet_count"]
    }

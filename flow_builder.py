import time
from collections import defaultdict

#main idea is that flow dictionary has a unique flow key for each flow and the flow is collected for 3 seconds
#first packet alwyas starts a flow which has a timespan of 3 seconds and any arrving packets are added to it the last packet after 3 seconds triggers
#the closing of the flow and pops the flow key and after summary the next packages open a new flow 

FLOW_TIMEOUT = 3# a window of 3 seconds if flow is longer than that its complete

flows = defaultdict(lambda: {   #stores active net flows 
    "start_time": None,         #default dict ensures that if a new flow key appears
    "packet_count": 0,          #it creates a new dict with same struct w/o throwing an error
    "total_bytes": 0,           #this is flag checking
    "packet_lengths": []        
})

#now we create a 5 tuple identifier for the net connectn 
#a flow is defined as ,source ip,dest ip,source port,destnation port and protocol TCP UDP
def get_flow_key(packet):
    try:
        if packet.haslayer("IP"):   #if a packet has ip header (bro i frogot)
            ip = packet["IP"]
            proto = ip.proto #packet's protocol header

            if packet.haslayer("TCP") or packet.haslayer("UDP"): #both have source and destination ports
                sport = packet.sport 
                dport = packet.dport
            else:
                sport = 0
                dport = 0 #for icmp cuz no port info

            return (ip.src, ip.dst, sport, dport, proto) #if tuple so when stored in the dict the values are less likely to change 

    except Exception:
        return None

    return None #if no return from try and except both

# packet handling eveyrthing has a flow key if its a new flow its current time is recorded as start_time
def update_flow(packet): #mandatory call for each packet
    key = get_flow_key(packet) #unique flow identifier checks which flow the packet belongs to 

    if key is None: #just skip if the key isnt there
        return None

    now = time.time() #current timestamp
    flow = flows[key]   

    if flow["start_time"] is None: #new flow
        flow["start_time"] = now

    pkt_len = len(packet)

    flow["packet_count"] += 1
    flow["total_bytes"] += pkt_len
    flow["packet_lengths"].append(pkt_len)

    if now - flow["start_time"] >= FLOW_TIMEOUT: #checking the timespan
        completed_flow = flows.pop(key)
        return build_features(completed_flow, key)

    return None


def build_features(flow, key):
    src, dst, sport, dport, proto = key #unpacking the featueres from the tuple

    packet_lengths = flow["packet_lengths"] #a list of all packet lengths in the flow

    return {
        "Destination Port": dport, #the main summary
        "Packet Length Mean": sum(packet_lengths) / len(packet_lengths),
        "Fwd Packet Length Max": max(packet_lengths),
        "Total Length of Fwd Packets": flow["total_bytes"],

        # used only for live rule, not ML model
        "Flow Packet Count": flow["packet_count"]
    }

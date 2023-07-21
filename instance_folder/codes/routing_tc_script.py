#!/opt/conda/envs/pytorch/bin/python3
import os
import sys
import subprocess as sp
import shlex

import warnings
warnings.filterwarnings("ignore")


from helper_functions import run_command, get_topology_info, get_tier_info


number_of_tiers, type_of_tiers, count_of_tiers, \
    source_tier_name, destination_tier_name, latency_between_tiers, bw_between_tiers = \
        get_topology_info('ins_folder/files/topology_file.txt')



current_tier_name, topology_name_list, topology_ip_list = \
    get_tier_info('ins_folder/files/tier_file.txt')

### get current tier private_ip and index
current_tier_ind = [idx for idx, s in enumerate(topology_name_list) if current_tier_name in s][0]
current_tier_ip = topology_ip_list[current_tier_ind]

### get the prev and next tiers and gws based on topology
next_gw = ""
prev_gw = ""
next_tiers_list = []
prev_tiers_list = []

if (current_tier_ind != 0):
    prev_gw = topology_ip_list[current_tier_ind-1]
if (current_tier_ind > 1):
    prev_tiers_list = topology_ip_list[:current_tier_ind-1]
if (current_tier_ind != len(topology_ip_list)-1):
    next_gw = topology_ip_list[current_tier_ind+1]
if (current_tier_ind < len(topology_ip_list)-2):
    next_tiers_list = topology_ip_list[current_tier_ind+2:]



### get interface name
res = run_command(f"ifconfig | grep -B 1 {current_tier_ip} | awk -F ':' 'NR==1{{print $1}}'")
interface_name = res[:-1]

##but first delete the prev rule if there's any
prev_route_dest = run_command("route -n | grep UGH | awk '{print $1}'")
prev_route_dest = [d for d in prev_route_dest.split('\n') if len(d)>1]
for d in prev_route_dest:
    if len(d)>1:
        run_command(f"sudo route del {d}")

### set ip routes between current node and gateways
if (next_gw != ""):
    run_command(f"sudo ip route add {next_gw} via {current_tier_ip} dev {interface_name}")
if (prev_gw != ""):
    run_command(f"sudo ip route add {prev_gw} via {current_tier_ip} dev {interface_name}")

### set ip routes for all prev and next tiers
for n in next_tiers_list:
    run_command(f"sudo ip route add {n} via {next_gw} dev {interface_name}")
for p in prev_tiers_list:
    run_command(f"sudo ip route add {p} via {prev_gw} dev {interface_name}")

run_command(f"sudo iptables -I FORWARD 1 -s 172.31.0.0/20 -i {interface_name} -d 172.31.0.0/20 -j ACCEPT")

### set tc commands for first and last tier (default qdisc)
## but first delete prev rules if there's any
prev_tc = run_command(f"tc qdisc show | grep root | grep refcnt | grep {interface_name}")
if (len(prev_tc) != 0):
    run_command(f"sudo tc qdisc del dev {interface_name} root")

if (current_tier_ind == 0):
    run_command(f"sudo tc qdisc add dev {interface_name} root netem delay {latency_between_tiers[0]}ms rate {bw_between_tiers[0]}mbit")

elif (current_tier_ind == len(topology_name_list)-1):
    run_command(f"sudo tc qdisc add dev {interface_name} root netem delay {latency_between_tiers[len(topology_name_list)-2]}ms rate {bw_between_tiers[len(topology_name_list)-2]}mbit")

### set tc commands for middle tiers (prio qdisc) + filtering
else:
    run_command(f"sudo tc qdisc add dev {interface_name} root handle 1: prio ")
    run_command(f"sudo tc qdisc add dev {interface_name} parent 1:1 handle 10: netem delay {latency_between_tiers[current_tier_ind]}ms rate {bw_between_tiers[current_tier_ind]}mbit")
    run_command(f"sudo tc qdisc add dev {interface_name} parent 1:2 handle 20: netem delay {latency_between_tiers[current_tier_ind-1]}ms rate {bw_between_tiers[current_tier_ind-1]}mbit")

    for n in list(next_tiers_list)+[next_gw]:
        if (n != ""):
            run_command(f"sudo tc filter add dev {interface_name} protocol ip parent 1:0 prio 1 u32 match ip dst {n} flowid 1:1")

    for p in list(prev_tiers_list)+[prev_gw]:
        if (p != ""):
            run_command(f"sudo tc filter add dev {interface_name} protocol ip parent 1:0 prio 2 u32 match ip dst {p} flowid 1:2")



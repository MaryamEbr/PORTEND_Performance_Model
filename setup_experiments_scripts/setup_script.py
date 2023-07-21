#!/Users/maryamebrahimi/.virtualenvs/ee_aws_env/bin/python3
import paramiko
import sys
import time
import numpy as np
import subprocess as sp

sys.path.append('ins_folder/codes')
from helper_functions import get_topology_info, describe_instance_cmd, launch_aws_instances, \
    start_aws_instances, wait_until_running, print_file, run_command, run_command_ssh, edit_ssh_config

tc_route_flag = False
file_flag = False
launch_flag = False
chrony_flag = False
if len(sys.argv) > 1:
    if 'tc' in sys.argv:
        tc_route_flag = True
    if 'file_transfer' in sys.argv:
        file_flag =True
    if 'launch' in sys.argv:
        launch_flag = True
    if 'chrony' in sys.argv:
        chrony_flag = True

# constant aws info
aws_region = "us-west-2"
aws_profile = "default"
key_name = "ee_key"


ami_dl = "ami-02ee8d5b35a4a74b2"
ami_amazon = "ami-0e33e5fdabd319577"
ami = [ami_amazon, ami_dl]

subnet = "subnet-004d623e94d51414a" #s-west-2a - Zone ID usw2-az2 - 172.31.0.0/20
security_group = "sg-01f25add4086ca917"


# topology/aws related info
number_of_tiers, type_of_tiers, count_of_tiers, source_tier_name,\
    destination_tier_name, latency_between_tiers, bw_between_tiers\
        = get_topology_info('ins_folder/files/topology_file.txt')
filter_name = 'mobile_topology5' # the name of the topology to filter instances by



# if launch flag is true-> launch 
if launch_flag == True:
    print("launching new instances...")
    launch_aws_instances(number_of_tiers, ami, count_of_tiers, type_of_tiers, subnet, security_group, key_name, filter_name)
    time.sleep(100)
    print("launched new instances")


# wait for samples to change state from nothing or stopped to running
wait_until_running(number_of_tiers, key_name, filter_name)
# time.sleep(50)  ##### uncomment this later

# update instance list after making sure they are running
describe_ins_list = describe_instance_cmd(key_name, filter_name)
print("instances are ready. the list:")
print(*describe_ins_list, sep='\n')


# edit ssh config file
edit_ssh_config(key_name, filter_name)
print("ssh config file is edited with new ip s")


# set up ssh and sftp to instances 
key = paramiko.RSAKey.from_private_key_file("ee_key.pem")
ssh_list = []
sftp_list = []

for i in range(len(describe_ins_list)):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    un="ec2-user" if i==0 else "ubuntu"
    ssh.connect(hostname=describe_ins_list[i]['public_ip'], username=un, pkey=key)
    sftp = ssh.open_sftp()
    ssh_list.append(ssh)
    sftp_list.append(sftp)


# run chrony commands
if chrony_flag == True:
    for i in range (len(describe_ins_list)):
        run_command_ssh(ssh_list[i], "sudo sed -i '1i server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4' /etc/chrony/chrony.conf")
        run_command_ssh(ssh_list[i], "sudo /etc/init.d/chrony restart")
        print("chrony is set up in tier", i)



# copy files to instances and make them executable
if file_flag == True or launch_flag == True:
    print("copying files to instances")

    # set up tier file (dynamic)
    for i in range(len(describe_ins_list)):
        tier_file = open('ins_folder/files/tier_file.txt', 'w')
        print_file(tier_file, 'current_tier_name > '+ describe_ins_list[i]['name'])
        for j in range(len(describe_ins_list)):
            print_file(tier_file, describe_ins_list[j]['name']+' >> '+describe_ins_list[j]['private_ip'])
        
        # zip, put, and unzip ins_folder (becuase put method only works with files not folders)
        run_command("rm -f ins_folder.zip")
        run_command("zip -r ins_folder.zip ins_folder/")
        time.sleep(5)
        run_command_ssh(ssh_list[i], "sudo rm -f -r ins_folder/")
        run_command_ssh(ssh_list[i], "sudo rm -f ins_folder.zip")
        time.sleep(5)
        un="ec2-user" if i==0 else "ubuntu"
        sftp_list[i].put('ins_folder.zip',f'/home/{un}/ins_folder.zip')
        time.sleep(5)
        run_command_ssh(ssh_list[i], 'sudo unzip -o ins_folder.zip')
        time.sleep(5)
        run_command_ssh(ssh_list[i], 'sudo chown ubuntu -R ins_folder')

        print("files are copied to tier", i)

        # make python files executable
        run_command_ssh(ssh_list[i], "chmod +x ins_folder/codes/routing_tc_script.py")
        run_command_ssh(ssh_list[i], "chmod +x ins_folder/codes/agent_source.py")
        run_command_ssh(ssh_list[i], "chmod +x ins_folder/codes/agent_inference.py")
        run_command_ssh(ssh_list[i], "chmod +x ins_folder/codes/agent_destination.py")

        print("python scripts are executable now in tier", i)


# tc ip route commands
if tc_route_flag == True:
    for i in range(len(describe_ins_list)):
        sftp_list[i].put("ins_folder/files/topology_file.txt", "ins_folder/files/topology_file.txt")
        # sftp_list[i].put("ins_folder/codes/routing_tc_script.py", "ins_folder/codes/routing_tc_script.py")
        run_command_ssh(ssh_list[i], "ins_folder/codes/./routing_tc_script.py")
        print("tc and ip route table are set in tier", i)
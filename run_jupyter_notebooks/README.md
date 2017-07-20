# Launching Jupyter notebooks

Here, you can find how to launch Jupyter notebooks (in the virtual environments you have created) in your local machine, in Piz Daint or in an AWS instance.

## Local
Just run `./launch_jupyter_local.sh`.

## Remote
To access a notebook running in a remote machine, you will need to establish an SSH tunnel with port forwarding.
Here, we will show how to configure dynamic port forwarding.

### Setting a Proxy

#### Installing a Proxy switcher
To proceed, it is recommended to add a proxy switcher in your browser, such as [this add-on](https://addons.mozilla.org/en-US/firefox/addon/proxy-switcher/) for Firefox.

#### Creating a SOCKS Proxy
In your browser, go to your Proxy settings (in your Proxy switcher if you have one), and choose a *Manual* configuration.
Do not set a HTTP Proxy, but set a SOCKS Proxy with:
```
Host: 127.0.0.1
Port: 10000
No Proxy for: localhost, 127.0.0.1
```
Where *10000* is just an arbitrary port number we have chosen.

You can also select `SOCKSv5` and enable `Remote DNS`.

Once you have set a SOCKS Proxy in your browser, you can proceed with establishing a connection with a remote machine.

### Piz Daint
Let USERNAME be your CSCS username.

1. Run `scp launch_jupyter_daint.sh USERNAME@daint.cscs.ch:`.
2. Run `ssh -D 10000 USERNAME@daint.cscs.ch`.
3. In Daint, run `sbatch launch_jupyter_daint.sh`.
This prints `Submitted batch job JJJJJJJ`, where JJJJJJJ is the id of your job.
4. Run `scontrol show job JJJJJJJ`. 
Once you have `JobState=RUNNING`, copy the ID of the node in which the notebook is running from the entry `NodeList=NODEID`.
5. Run `less JUPYTER_OUT.JJJJJJJ.log`, where JUPYTER\_OUT is the output filename set in `launch_jupyter_daint.sh`.
4. Copy the URL where the notebook is available to your local browser and substitute the *0.0.0.0* IP address with NODEID.

### AWS
Let INSTANCE_IP be the IP address of your AWS instance.

1. Run `scp launch_jupyter_aws.sh ubuntu@INSTANCE_IP:`.
2. Run `ssh -D 10000 ubuntu@INSTANCE_IP`.
3. In the instance, run `./launch_jupyter_aws.sh`.
4. Copy the URL where the notebook is available to your local browser and substitute the *0.0.0.0* IP address with INSTANCE_IP.
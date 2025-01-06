# Automation scripts  

This directory contains scripts that help automate running and managing a large number of experiments

1. Local experiment automation using `byobu` session management. (requries `byobu` to be installed)

   ```bash
    ./run_experiments-byobu.sh
   ```

2. Cloud deployment using kubernetes jobs. (requires `kubectl` and `helm` to be installed)

    ```bash
    ./launch_k8s_jobs.sh
    ```

>[!NOTE]
>
> The helm chart is located in the `chart` directory. The chart is used to deploy the experiment container to the kubernetes cluster. The chart is used by the `launch_k8s_jobs.sh` script.  
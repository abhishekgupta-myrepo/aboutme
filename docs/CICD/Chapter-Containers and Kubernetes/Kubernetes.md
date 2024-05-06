# Simple Kubernetes Project
## Kubernetes Background 
- A Kubernetes cluster can be deployed and tested locally following the steps mentioned below - 
	- This simplicity must not be confused with the actual complexity and the depth of the functionality that Kubernetes provides for running production clusters.
	- Kubernetes is currently far and wide an industry standard. This is generally used for running 10000+ nodes clusters.
	- It is now an open source project originally developed by google.
	- Mesos is the nearest competitor that is used by many giants for running their production environment.

## Steps to Run Kubernetes Cluster Locally
- **Step 1** : Install Kubectl from the website for the local deployment of the cluster
	- [Getting started | Kubernetes](https://kubernetes.io/docs/setup/)
	![300](kubectl1.png)

- **Step 2** : Install Kind or Minikube which is used for running deployments in local or in development environment. Both the options are provided on the Kubernetes website. [minikube start | minikube (k8s.io)](https://minikube.sigs.k8s.io/docs/start/)
	-  Run the command ==minikube start== . Note this must be run from the ==command prompt==
		- Ensure that the .exe file that you downloaded for the minikube is set in the PATH environment variable.
		- Ensure that prior to starting the minikube cluster you have installed ==kubectl== 
			- The steps to install kubectl [which the command line for Kubernetes] are similar to installing the minikube. i.e. once the .exe file for kubectl has been downloaded setup the ==PATH== variable for kubectl.exe file.

![500](kubectl2.png)

- **Step 3** : Run this command  to get information about the cluster ==kubectl cluster-info==
- **Step 4**: Run this command to deploy the container image in the minikube cluster. ==kubectl create deployment nodeapplication2 -- image=node ==
	- Note: Minikube pulls the image provided either from Dockerhub by default or from the specified repo that can be mentioned as the full alternate url to the image.
	- Unlike docker swarm that pulls the images and the services stack from the docker compose .yml minikube pulls images directly that have been pushed into the repo.
	- So in case you need test the local cluster deployment you will need to push the image to Docker hub repo from where it will be pulled.
- **Step 5** : Run this command to check the deployment ==kubectl get deployments==

 To develop a CI pipeline from Github project using Travis CI app refer to the notes on [[CI with Travis]]
	- This takes it to full circle where once the code is developed, tested locally, and committed to the git repo in Github a continuous integration tool such as Travis can be used for continuous build that gets triggered for each commit to compile the code and push the image into a remote repository such as docker hub.
# Docker Compose
Instructions below are based on following the directions provided during the course :  Docker for developers by Emmanuel Henri - [Docker for developers (linkedin.com)](https://www.linkedin.com/learning/docker-for-developers-14493163/docker-for-developers?autoplay=true&resume=false)

---
## Step by Step Guide to Run Docker Compose
- Follow the steps below to develop a full stack service comprising of multiple containers.
- The instructions for developing a full stack are written the docker compose .yml file. Look at the sample project developed available at the GitHub  - [abhishekgupta-myrepo/docker_poc (github.com)](https://github.com/abhishekgupta-myrepo/docker_poc)
- Once the docker compose file is written preform the following steps to create a service.
- **Step 1** : Run the below command to create images for all the services that form part of the full stack and must be deployed as a single unit.
```bash
PS C:\Users\abhis\Projects\docker_poc> docker-compose build
[+] Building 22.1s (11/11) FINISHED
 => [internal] load build definition from Dockerfile                                                      0.0s
 => => transferring dockerfile: 155B                                                                      0.0s
 => [internal] load .dockerignore                                                                         0.0s
 => => transferring context: 66B                                                                          0.0s
 => [internal] load metadata for docker.io/library/node:latest                                            1.7s
 => [auth] library/node:pull token for registry-1.docker.io                                               0.0s
 => [1/5] FROM docker.io/library/node@sha256:8a45c95c328809e7e10e8c9ed5bf8374620d62e52de1df7ef8e71a9596e  0.0s
 => [internal] load build context                                                                         0.3s
 => => transferring context: 16.08MB                                                                      0.3s
 => CACHED [2/5] WORKDIR /usr/src/app                                                                     0.0s
 => [3/5] COPY package*.json ./                                                                           0.1s
 => [4/5] RUN npm install                                                                                18.3s
 => [5/5] COPY . .                                                                                        0.1s
 => exporting to image                                                                                    1.4s
 => => exporting layers                                                                                   1.4s
 => => writing image sha256:3b656d9f2fa551b00589d89b61f563f07c33c7f8eb2a2ce191a7b6d2d6c14576              0.0s
 => => naming to docker.io/library/docker_poc-app                                                         0.0s

Use 'docker scan' to run Snyk tests against images to find vulnerabilities and learn how to fix them
```

- **Step 2** : Run the following command to spin up specific container to be created as part of the full stack. 
	- Note: Even if you bring up each container one at a time even then all the containers form part of the stack. 
	- Make sure to trigger the containers in the order so that the other container does not fail that is dependent upon the other container for successful boot. This is done by mentioning the specific container to kickoff
	-  Reason for specifying the -d and mentioning the container that needs to start first is that docker compose will start all the containers in parallel and will run into issues if the database is not up.
```bash 
PS C:\Users\abhis\Projects\docker_poc> docker-compose up -d mongo
[+] Running 10/10
 - mongo Pulled                                                                                          19.8s
   - 675920708c8b Pull complete                                                                           5.8s
   - 6f9c8c301e0f Pull complete                                                                           5.9s
   - 73738965c4ce Pull complete                                                                           6.3s
   - 7fd6635b9ddf Pull complete                                                                           6.9s
   - 73a471eaa4ad Pull complete                                                                           7.0s
   - bcf274af89b0 Pull complete                                                                           7.1s
   - 04fc489f2a3b Pull complete                                                                           7.2s
   - 6eff8a505292 Pull complete                                                                          18.6s
   - a5ef4431fce7 Pull complete                                                                          18.6s
[+] Running 2/2
 - Network docker_poc_default  Created                                                                    0.0s
 - Container mongo             Started                                                                    1.1s
PS C:\Users\abhis\Projects\docker_poc> 
```

- **Step 3** : Run the below command to look into the logs of specific container.
	- This is not a core part of the sequence of instructions to spin up this stack and is just to validate the state of the containers. 
```bash
PS C:\Users\abhis\Projects\docker_poc> docker logs e74cd
{"t":{"$date":"2022-09-15T20:59:49.287+00:00"},"s":"I",  "c":"NETWORK",  "id":4915701, "ctx":"-","msg":"Initialized wire specification","attr":{"spec":{"incomingExternalClient":{"minWireVersion":0,"maxWireVersion":17},"incomingInternalClient":{"minWireVersion":0,"maxWireVersion":17},"outgoing":{"minWireVersion":6,"maxWireVersion":17},"isInternalClient":true}}}

```
- **Step 4** : Run the below command to stop all the containers that form part of a stack built through the docker compose.
	- Stopping all the containers can be done by a single command
	- Make sure to stop the ==docker compose stop== to stop the service else if you stop the containers individually they will spin up again.
```bash
PS C:\Users\abhis\Projects\docker_poc> docker-compose stop
[+] Running 2/2
 - Container app    Stopped                                                                               1.0s 
 - Container mongo  Stopped 
```

- Once the docker compose is ready for deployment it can be used for deployment through container orchestration tools such as - 
	- Docker Swarm: look at the notes @ [[Docker-Swarm]]. This is native to Docker and comes installed with docker desktop.
	- Kubernetes: look at the notes @[[Kubernetes]]. It is the current industry standard. Instead of setting up your own Kubernetes cluster most common practice is to use it as a managed services through cloud providers such as GCP or AWS.
	- Mesos


---
- These notes are based on my understanding of the topic. Some of the content and images refers to the online articles, learning courses I attended. I encourage you to lookup the links provided to develop your own understanding of it at deeper level.
- The intention of these notes is to serve as reference marker to other sites, online courses, books that I found useful while learning about it.
- If you would like to share any thoughts or give comments please feel free to reach me @ abhishekgupta86@gmail.com or @ www.linkedin.com/in/abhishekgupta86
---


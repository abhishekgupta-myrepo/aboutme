# Compile simple project in docker
Instructions below are based on following the directions provided during the course :  Docker for developers by Emmanuel Henri - [Docker for developers (linkedin.com)](https://www.linkedin.com/learning/docker-for-developers-14493163/docker-for-developers?autoplay=true&resume=false)

---
## Step to Create Docker based Project
Follow the below steps to setup a simple backend container based docker project.

- **Step 1** - Create the Dockerfile. Refer to the GitHub project - [abhishekgupta-myrepo/docker_poc (github.com)](https://github.com/abhishekgupta-myrepo/docker_poc)
	- The backend project in within the /app folder in the git project.
	- To run and compile the docker image, go to the app folder in the Visual studio code PowerShell terminal to type in the commands below - 
	- Before you run the commands make sure you have the docker desktop installed. Follow the steps mentioned in [[About Docker]]
	- Make sure you have installed the VS code extension as it helps in suggesting the commands when writing Dockerfile.
	- All the commands are uppercase
- **Step 2** - Run the docker command to build the image - 
```bash
PS C:\Users\abhis\Projects\docker_poc> docker --version
Docker version 20.10.17, build 100c701

PS C:\Users\abhis\Projects\docker_poc> docker build -t simplebackend .
[+] Building 51.1s (11/11) FINISHED
 => [internal] load build definition from Dockerfile                                                                      0.1s
 => => transferring dockerfile: 155B                                                                                      0.0s
 => [internal] load .dockerignore                                                                                         0.0s
 => => transferring context: 66B                                                                                          0.0s
 => [internal] load metadata for docker.io/library/node:latest                                                            2.1s
 => [auth] library/node:pull token for registry-1.docker.io                                                               0.0s
 => [internal] load build context                                                                                         0.1s
 => => transferring context: 30.39kB                                                                                      0.0s 
 => [1/5] FROM docker.io/library/node@sha256:8a45c95c328809e7e10e8c9ed5bf8374620d62e52de1df7ef8e71a9596ec8676            30.5s 
 => => resolve 
 => => extracting sha256:8471b75885efc7790a16be5328e3b368567b76a60fc3feabd6869c15e175ee05                                 4.7s 
...... Multiple lines of progress......
 => => extracting sha256:b51256989aaaa6444b004b6f1dc3753c7c5a0f2132187622ee395e1327c36061                                 0.0s 
 => [2/5] WORKDIR /usr/src/app                                                                                            0.5s 
 => [3/5] COPY package*.json ./                                                                                           0.0s 
 => [4/5] RUN npm install                                                                                                16.4s 
 => [5/5] COPY . .                                                                                                        0.0s 
 => exporting to image                                                                                                    1.4s 
 => => exporting layers                                                                                                   1.4s 
 => => writing image sha256:27a88271fc4f9634fe0a96e8e79ad533303f454f01966daa78fa5562931c4760                              0.0s 
 => => naming to docker.io/library/simplebackend                                                                          0.0s 

Use 'docker scan' to run Snyk tests against images to find vulnerabilities and learn how to fix them
PS C:\Users\abhis\Projects\docker_poc>

```
- **Step 3** - Run the following docker command to check the image built locally- 
```bash
PS C:\Users\abhis\Projects\docker_poc> docker images
REPOSITORY      TAG       IMAGE ID       CREATED         SIZE
simplebackend   latest    27a88271fc4f   6 minutes ago   1.05GB
alpine/git      latest    692618a0d74d   2 weeks ago     43.4MB
```
- **Step 4** - Run the following docker command to instantiate container based on the image that you just created in the previous step-
	- Mention about the port using -p and bind the container port to the local machine/VM port as show in the command below.
	- Refer to the image using the tag that you assigned during the build process.
```bash
PS C:\Users\abhis\Projects\docker_poc> docker run -p 4000:4000 simplebackend

> backend@1.0.0 start
> nodemon --exec babel-node index.js

[nodemon] 2.0.19
[nodemon] to restart at any time, enter `rs`
[nodemon] watching path(s): *.*
[nodemon] watching extensions: js,mjs,json
[nodemon] starting `babel-node index.js`
Your server is running on PORT: 4000
```
- **Step 5** - Run the following docker command to check if the container is up and running as a service
	- The container running can also be checked from the docker desktop as well as the number of images that have been created.
	- Make sure that the container that you are running you shut down using the ==stop command==
	- Also, keep a check on the number of images that have been created. These images general run into 1-2 GBs and will very quickly eat up your local disk space.
	
```bash
PS C:\Users\abhis\Projects\docker_poc> docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS          PORTS                    NAMES
cd637f86b4a8   simplebackend   "docker-entrypoint.s…"   33 seconds ago   Up 32 seconds   0.0.0.0:4000->4000/tcp   great_heyrovsky
PS C:\Users\abhis\Projects\docker_poc> 
PS C:\Users\abhis\Projects\docker_poc> docker stop cd637
cd637
```

- For running a full stack that comprise of frontend , backend, and a database consists of spinning up multiple containers that often form a stack. These containers can be combined to create a service that can be built suing ==docker compose==
- To look at how to setup a multi-container based full stack service look at the notes [[Docker Compose]]
- To develop a CI pipeline from Github project using Travis CI app refer to the notes on [[CI with Travis]]
	- This takes it to full circle where once the code is developed, tested locally, and committed to the git repo in Github a continuous integration tool such as Travis can be used for continuous build that gets triggered for each commit to compile the code and push the image into a remote repository such as docker hub.


---
- These notes are based on my understanding of the topic. Some of the content and images refers to the online articles, learning courses I attended. I encourage you to lookup the links provided to develop your own understanding of it at deeper level.
- The intention of these notes is to serve as reference marker to other sites, online courses, books that I found useful while learning about it.
- If you would like to share any thoughts or give comments please feel free to reach me @ abhishekgupta86@gmail.com or @ www.linkedin.com/in/abhishekgupta86
---


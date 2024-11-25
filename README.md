## 1.Pod

<!-- !![image](https://user-images.githubusercontent.com/74223025/227868417-dc461649-ee79-4dcc-9cf5-197c8b56f603.png) -->

<img src="https://user-images.githubusercontent.com/74223025/227868417-dc461649-ee79-4dcc-9cf5-197c8b56f603.png" width="600" height="300">


## 2. Service

Service has 2 functioanlities:
1. permanent IP address: gives permanent IP to pods.
2. load balancer : it will forward the request to pod which is free.

<!-- !![image](https://user-images.githubusercontent.com/74223025/227868862-2e0525ac-d3a2-40be-998a-ab57d12d8b18.png) -->

<img src="https://user-images.githubusercontent.com/74223025/227868862-2e0525ac-d3a2-40be-998a-ab57d12d8b18.png" width="600" height="300">

## 3. Ingress

![image](https://user-images.githubusercontent.com/74223025/227881173-4152897b-06a3-4fad-8881-53e7afe7d556.png)

          
Now if you want to connect with your application through browser then you can use URL like: http://ServerIp:NodePort. Here external request will first come to ingress and then ingress will route it to different pods clusters(ClusterIP).

<!-- !![image](https://user-images.githubusercontent.com/74223025/227869658-12101089-fe07-45a5-95e8-54f0ee4d7547.png) -->
<img src="https://user-images.githubusercontent.com/74223025/227869658-12101089-fe07-45a5-95e8-54f0ee4d7547.png" width="600" height="300">

## 4. ConfigMap
As we know pods communicates with each other using service IP addresses however where will you store those IP addresses? => ConfigMap. It stores the data into plain text formats.

![image](https://user-images.githubusercontent.com/74223025/229040796-6725af3f-bf25-448d-8c1d-522099afeae1.png)

## 5. Secret
Dont put credentials into ConfigMap, Secrets are used to store sensitive information like username and password. It stores the data into encoded format.

![image](https://user-images.githubusercontent.com/74223025/229041019-b1739275-7549-42c1-8916-a8b7c5ec5c9c.png)

## 6. Volumes
If pod gets restarted then data associated with pod also get deleted. Hence <b>Volumes</b> attaches a physical storage to pods.
Volumes help pods for data persistency.

![image](https://user-images.githubusercontent.com/74223025/229419102-b41e265b-7b7e-416a-afb1-70e9034407e2.png)

## 7. Deployments (StateLess apps)
- it is replicating mechanism for StateLess apps
- when you want to create a replica of pods, you are not really creating another pods instead you will <b>define blueprint of pods</b> and specify how many replica of that pod. The <b>blueprint</b> is called as Deployment.
- It tells Kubernetes how to create or modify instances of the pods that hold a containerized application.
- efficiently scale the number of replica pods,
- enable the rollout of updated code in a controlled manner,or roll back to an earlier deployment version if necessary.

![image](https://user-images.githubusercontent.com/74223025/229445639-4d742588-aaf0-43ad-9a0c-5ff699a9f55e.png)

## 8. Statefulset (StateFul apps/database)
- it is replicating mechanism for StateFul apps/database
- We cant replicate DATABASE using <b>Deployments</b>, reason database has state means data. Meaning taht if we have replicas of database they all should access same shared data storage. Hence there we need some mechanism that manages which pods are currently writting data to data storage or which pods are currently reading data from data storage, <b>in order to avoid Data Incosistency</b> (database read-write synchronization)
- that mechanism in addition to replicating the pods is offered by Statefulset
- StateFul apps/databases should be created using Statefulset not by Deployments.
- deploying datavase with Statefulset is tedious work hence most of the time databases are hosted outside the Kubernetes clusters.

![image](https://user-images.githubusercontent.com/74223025/229447848-3aa31188-4a2c-41ca-ba1b-52159377846b.png)


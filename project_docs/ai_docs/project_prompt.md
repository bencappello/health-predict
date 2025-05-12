# Project-2: Health Predict - Predicting Patient Readmission

## Health Predict: Problem & Objective

### Business Problem
Navigating the intricate landscape of healthcare, the challenge of patient readmission emerges as a critical concern that demands innovative solutions. Patient readmission, or the return of a patient to the hospital shortly after discharge, not only poses a significant financial burden on healthcare systems but also underscores potential gaps in patient care. High rates of readmission often signal unresolved health issues, insufficient post-discharge support, or inadequate coordination among healthcare providers. This recurrent cycle not only impacts the well-being of patients but also places additional strain on healthcare resources.

Addressing the problem of patient readmission is not merely about reducing costs; it is fundamentally rooted in a commitment to enhancing the overall quality of patient care, ensuring seamless transitions between hospital and home, and ultimately fostering a healthcare system that prioritizes sustained well-being and optimal recovery.

### Technical Problem
With Health Predict, our aim is to design a predictive model to assess and anticipate the likelihood of patient readmission.

Health Predict should analyze a myriad of patient-specific factors, such as medical history, demographic information, treatment protocols, post-discharge follow-ups etc. to generate a comprehensive prediction model. The system aims to proactively identify individuals at a higher risk of readmission, allowing healthcare providers to implement targeted interventions and personalized care plans.

In the context of a predictive model like Health Predict, MLOps (Machine Learning Operations) is crucial for several reasons:

* MLOps ensures the seamless integration of machine learning models into the healthcare workflow. By establishing robust deployment pipelines and monitoring mechanisms, MLOps facilitates the efficient transition of predictive models from development to production, enabling healthcare professionals to leverage these models in real-world scenarios.
* Additionally, MLOps contributes to model performance monitoring and management. In the dynamic healthcare environment, patient conditions and treatment protocols evolve. MLOps enables continuous monitoring of model performance, allowing for timely updates and refinements to ensure the predictive accuracy of the model aligns with the evolving healthcare landscape.

Therefore, this challenge is designed to evaluate your proficiency in the complete Machine Learning (ML) lifecycle, extending beyond conventional model development. In addition to assessing your modeling skills, this problem aims to gauge your understanding and application of MLOps practices throughout the ML workflow.

This holistic evaluation seeks to measure your competency in handling the end-to-end ML process, incorporating MLOps principles to guarantee the model's reliability, scalability, security, and compliance with ethical standards.

### Objectives
* Examine data and conduct feature engineering on datasets related to patient readmission.
* Develop and train machine/deep learning models for forecasting patient readmission.
* Orchestrate model training and optimize hyperparameters using an orchestration system like Airflow. Execute parallel, large-scale experiments with an HPO tool such as RayTune.
* Establish and log all experiments in MLflow for comprehensive tracking.
* Retrieve the best-performing model and deploy it as a RESTful API endpoint.
* Automate packaging, and deployment processes using CI/CD tools like Jenkins or orchestration tools such as Airflow.
* Implement a model monitoring system using tools like Evidently to detect potential drift.
* Develop orchestration pipelines to retrain and deploy models based on drift detection.

Please note that the mentioned tools are provided for reference based on the curriculum. Feel free to explore alternative tools that align with the project's goals.

The dataset can be downloaded from [https://www.kaggle.com/datasets/brandao/diabetes](https://www.kaggle.com/datasets/brandao/diabetes). Use the data as per your convenience, splitting into Test/Train, or for simulating data drifts.

## Health Predict: Deliverables/Submission Guidelines

### Experimentation Notebook
Submit the first series of experiments in the form of a Jupyter Notebook.
The notebook should include initial Exploratory Data Analysis, Feature Engineering, Model Exploration, and so forth.

### MLOps Specific
This project specifically aims to evaluate your comprehension of the various concepts covered in the MLOps modules of the course.
Consequently, your project must integrate most, if not all, of these concepts, encompassing Data Management, Large Scale Model Training, Experiment Tracking, Model Packaging, Deployment, Inference, Model Monitoring and Continuous Training in Production.
An optimal submission should also orchestrate these processes using an Orchestration engine, such as Airflow.

### Documentation
Ensure that the code is thoroughly documented, explaining each module/aspect clearly.
The documentation should incorporate snapshots of the executed code or pipeline, and optionally a video.
Additionally, furnish a comprehensive user manual detailing the execution process of your code or pipelines, including instructions for setting up the required tools and dependencies.
The documentation should also explain the System Architecture and provide additional implementation details.

### Subsequent Enhancements
A project is never perfect, especially in Machine Learning. With a limited submission timeframe, and a vast array of tools to work with, implies that perfection is a challenging goal.
We acknowledge this consideration and recommend including a section that outlines potential future enhancements for the project, providing detailed elaboration on all aspects.

## Health Predict: Setup Suggestions

Given that this project may require you to set up and manage many ML tools, it becomes important to ensure that those tools communicate well with each other.

One major hurdle that you may face is while setting up Airflow, if you choose to use it. Since Airflow consists of a number of components (scheduler, worker, web server, databases, etc.), the best way to manage it is using Docker Compose, similar to how we did in Classroom Demos and Assignment. That said, in real-life production systems, the Docker Compose methodology is not suitable for production scenarios.

It is crucial for you to understand that a Docker image has a file system and network which is segregated from the host. In order for Docker to access the host's file system, you would need to mount host directories to the containers in the Docker Compose file. Recall how this was done in the assignments and classroom demos for various MLOps classes involving Airflow.

Also, since Docker network is isolated from the host machine, any tool that you deploy separately on the host machine wouldn’t be directly accessible inside Airflow and/or other containers. There are several ways to handle this, but the most effective way being that the tools that you deploy on your hosts are exposed to the internet. That way they can be accessed easily inside Docker containers as well.

That said, you are free to explore alternative forms of deployment, including standalone deployment (without Docker involved) of all tools, or deploying everything within the same Docker network, using a single Docker Compose. You are also free to explore other MLOps tools which were not taught in the course.

Another crucial factor to consider is the consistent use of data transformers, such as CountVectorizer or OneHotEncoder, throughout the entire life cycle of your machine learning model. It's essential to refrain from fitting these transformers on production data during model inference, as doing so could result in data leakage and compromise the model's reliability. To address this, it's recommended to devise innovative solutions that allow the seamless utilization of these transformation objects in production without the need to fit them on test data.

## Health Predict: Project Milestones

### Data Exploration and Preprocessing
* Download and explore the dataset from the provided link.
* Conduct initial Exploratory Data Analysis (EDA) to understand data characteristics.
* Preprocess the data, handle missing values, and perform necessary transformations.
* Identify relevant features for predicting patient readmission.
* Perform feature engineering to enhance the predictive power of the model.
* Use appropriate techniques for feature selection if needed.

### MLOps Tooling Setup
* Deploy all the tools required for MLOps best practices.

### Model Development and Training
* Choose suitable machine/deep learning models for patient readmission prediction.
* Split the dataset into training and testing sets.
* Train and validate the models, optimizing hyperparameters for better performance.

### API Development Deployment
* Retrieve the best-performing model.
* Package the model as a Docker Image and deploy on Kubernetes if required as a REST API Endpoint.
* Automate packaging and deployment processes using CI/CD tools.

### Model Monitoring
* Implement a model monitoring and drift detection system using tools like Evidently.

### Orchestration and Optimization
* Orchestrate the entire model training, evaluation, deployment and retraining pipeline using an Orchestration System such as Airflow.

### Documentation and User Manual
* Thoroughly document the code, explaining each module and aspect clearly.
* Include snapshots and possibly a video in the documentation.
* Provide a comprehensive user manual detailing the execution process and tool dependencies.

### Submission
* Ensure all deliverables are prepared, including the experimentation notebook, MLOps integration, and documentation.
* Submit the project for evaluation, adhering to the submission guidelines.

## Health Predict: Evaluation Criteria

* **EDA and Data Preparation (Weightage: 5%)**
    Demonstrate an understanding of the dataset through thorough Exploratory Data Analysis (EDA). Offer insights into key features and patterns. Perform Data Preparation, including thorough cleaning, transformation, and effective data structuring. Implement optimal preprocessing and feature engineering techniques to establish a foundation for subsequent stages in the machine learning workflow.
* **Large Scale Hyperparameter Optimization and Experiment Tracking (Weightage: 15%)**
    Conduct large-scale hyperparameter tuning efficiently utilizing RayTune or an equivalent tool. Implement comprehensive experiment tracking, maintaining detailed records of hyperparameters, metrics, configurations, models, and data transformation artifacts in MLflow or a similar tool.
* **Model Packaging and Deployment (Weightage: 15%)**
    Develop robust inference APIs with a focus on seamless integration and standardized model deployment. Provide API documentation accessible through Swagger or an equivalent tool. Ensure consistency in data transformers between the Inference API and the ones used during training. Package models as Docker images to ensure portability and reproducibility. Deploy the Docker image on a Kubernetes cluster for effective deployment.
* **Model Monitoring (Weightage: 15%)**
    Create a comprehensive monitoring dashboard to track the model's performance in production. Implement monitoring for data and model drift using Evidently or an equivalent tool. Conduct data drift detection on a diverse set of columns using various drift detection measures like PSI, KS Test, etc.
* **Continuous Integration and Deployment (Weightage: 15%)**
    Automate the Model/API Build and Deployment process using Jenkins or an equivalent tool. Configure Build and Deployment Jobs to reference the inference code located on a GitHub repository or a local server for Docker image generation, followed by deployment on Kubernetes. Enable remote triggering of Build and Deployment Jobs for efficient remote orchestration.
* **Orchestration (Weightage: 20%)**
    Implement the orchestration of Data Preparation, Hyperparameter Optimization, Model Packaging, and Model Deployment processes using Airflow or an equivalent tool. Establish remote integration with build tools like Jenkins or equivalent. Design the orchestration pipeline to periodically assess if a Data Drift has occurred with newly available data, enabling it to make informed decisions on whether to initiate model retraining.
* **Solution Documentation (Weightage: 15%)**
    Thoroughly document the entire implementation, providing clear instructions on setting up operational tools and rerunning the entire pipeline or individual tasks. Ensure the documentation includes a comprehensive system architecture of the orchestration pipeline, elucidating the functionality of each step in detail.

## Health Predict: FAQs

* **Q. Is it advisable to dedicate extensive time to enhance my model's accuracy in this project?**
    A. While acknowledging the importance of machine learning, it's crucial to note that the primary emphasis of this project lies in constructing the MLOps ecosystem. Although you have the flexibility to enhance the statistical performance of your model, please be aware that your submission won't be assessed based on those improvements.
* **Q. Can I use other tools than the ones suggested?**
    A. The project suggests tools like Airflow, Jenkins, and Evidently, but participants are encouraged to explore alternative tools that align with the project's goals.
* **Q. Why is consistent use of data transformers crucial in the machine learning model life cycle?**
    A. Consistent use of data transformers, such as CountVectorizer or OneHotEncoder, prevents data leakage and ensures the model's reliability.
* **Q. Is a GPU-based machine necessary for conducting my experiments?**
    A. As previously emphasized, the main focus is not on the statistical performance of the model. Therefore, it is not a requirement for your experiments. In case you do require a GPU instance, please try to conduct your experiments on Kaggle, which offers a 35 hours of multi GPU compute per week. In case you feel that it’s not sufficient, do let the instructor know.

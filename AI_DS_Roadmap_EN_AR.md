# AI & Data Science — Complete Learning Roadmap (README-ready)

> An exhaustive, well-structured learning roadmap you can publish as a GitHub `README.md`. Includes a full English version followed by a complete Arabic translation. Designed to help beginners through advanced learners build practical skills, projects, and a professional portfolio.

---

## Table of Contents
1. [Project Goal & Audience](#project-goal--audience)
2. [How to use this roadmap](#how-to-use-this-roadmap)
3. [High-level Phases](#high-level-phases)
4. [Phase-by-phase Detailed Curriculum](#phase-by-phase-detailed-curriculum)
   - Environment & Essentials
   - Programming & Data Foundations
   - Mathematics for Machine Learning
   - Classical Machine Learning
   - Deep Learning & Modern Architectures
   - AI Engineering & Model Serving
   - MLOps & Production Lifecycle
   - Big Data & Streaming
   - Specializations
   - Research & Advanced Topics
5. [Portfolio Projects & Deliverables]
6. [Recommended Repo Structure & Templates]
7. [README & Repository Polishing Tips]
8. [License & Contribution]
9. [Checklist]

---

## Project Goal & Audience
**Goal:** Create a single, detailed roadmap that teaches everything needed to become a professional in Data Science and AI engineering — from environment setup and mathematics to production deployment and research. The file is written in clear, structured English and followed by a full Arabic translation so it can be published as a bilingual GitHub README.

**Audience:** Students, self-taught engineers, early-career ML/AI engineers, and hobbyists.

---

## How to use this roadmap
1. Start with **Environment & Essentials** and **Programming & Data Foundations**. Build small projects as you learn.
2. Move to **Mathematics** while applying math concepts to projects (PCA, gradient descent, etc.).
3. Progress through **Classical ML → Deep Learning → Engineering → MLOps**.
4. Pick specializations (NLP, CV, Recommenders) after finishing core phases.
5. For each topic: learn the theory, follow a hands-on tutorial, and produce a reproducible project with clean notebooks and production-ready code.

---

## High-level Phases
- **Phase 0 — Environment & Essentials**
- **Phase 1 — Programming & Data Foundations**
- **Phase 2 — Mathematics for ML**
- **Phase 3 — Classical Machine Learning**
- **Phase 4 — Deep Learning & Modern Architectures**
- **Phase 5 — AI Engineering & Model Serving**
- **Phase 6 — MLOps & Production Lifecycle**
- **Phase 7 — Big Data & Streaming**
- **Phase 8 — Specializations**
- **Phase 9 — Research & Advanced Production**

---

## Phase-by-phase Detailed Curriculum

### Phase 0 — Environment & Essentials
**Skills:**
- Install and manage Python environments (venv, conda)
- Basic shell commands, SSH, file management
- Git and GitHub workflow (branches, PRs, issues)

**Tools:** Python 3.9+, Conda/venv, Git, VSCode, JupyterLab

**Deliverables:** `environment.yml` or `requirements.txt`, a simple README, first commit on GitHub

---

### Phase 1 — Programming & Data Foundations
**Skills:**
- Python core: data structures, OOP, modules, virtual environments
- pandas and NumPy for data processing
- Data visualization basics: matplotlib, seaborn, plotly
- SQL for data extraction and joins

**Tools/Libraries:** pandas, NumPy, matplotlib, seaborn, plotly, Jupyter

**Suggested Projects:**
- EDA report (Titanic or public dataset)
- Data cleaning pipeline and reproducible notebook

---

### Phase 2 — Mathematics for Machine Learning
**Topics to master:**
- Linear algebra: vectors, matrices, eigenvalues, SVD
- Probability & statistics: distributions, expectation, hypothesis testing
- Calculus & optimization: derivatives, gradients, chain rule, gradient descent
- Matrix calculus basics used in backpropagation

**How to practice:** Implement PCA, linear regression, and gradient descent from scratch using NumPy and visualize results.

---

### Phase 3 — Classical Machine Learning
**Skills:**
- Supervised algorithms: linear/logistic regression, decision trees, ensemble methods (Random Forest, XGBoost)
- Unsupervised learning: K-means, hierarchical clustering
- Model evaluation: cross-validation, metrics, bias-variance tradeoff
- Feature engineering and preprocessing

**Tools:** scikit-learn, XGBoost, LightGBM

**Projects:**
- Tabular classification pipeline (with feature engineering and model explainability using SHAP)
- Regression benchmark (housing prices)

---

### Phase 4 — Deep Learning & Modern Architectures
**Skills:**
- Tensors, autograd, training loops, optimization techniques
- CNNs for vision, RNNs/LSTM for sequences, Transformers for text
- Transfer learning and fine-tuning
- Regularization techniques (dropout, weight decay, batch normalization)

**Tools:** PyTorch (recommended), TensorFlow/Keras, Hugging Face Transformers

**Projects:**
- Build and train a CNN on CIFAR-10
- Fine-tune a transformer for sentiment analysis or question-answering

---

### Phase 5 — AI Engineering & Model Serving
**Skills:**
- Convert research notebooks into production modules
- Build inference APIs (FastAPI), containerize with Docker
- Model formats: ONNX, TorchScript

**Tools:** FastAPI, Docker, ONNX, TorchServe, TensorFlow Serving

**Projects:**
- Serve a trained classifier via FastAPI + Docker
- Convert a PyTorch model → ONNX and benchmark

---

### Phase 6 — MLOps & Production Lifecycle
**Skills:**
- Experiment tracking and reproducibility
- Data versioning and pipelines
- CI/CD for ML models, monitoring and alerts

**Tools:** MLflow, DVC, GitHub Actions, Airflow, Prometheus/Grafana

**Projects:**
- Create a training pipeline with DVC and log experiments with MLflow
- Build a CI workflow that runs tests and model validation

---

### Phase 7 — Big Data & Streaming
**Skills:**
- Distributed data processing and ETL
- Real-time ingestion and stream processing

**Tools:** Apache Spark (PySpark), Kafka, Parquet

**Projects:**
- ETL pipeline in Spark that processes large simulated data and stores Parquet outputs
- Kafka consumer that feeds a real-time feature store

---

### Phase 8 — Specializations (choose as needed)
**Natural Language Processing (NLP):** tokenization, embeddings, transformers, RAG (retrieval-augmented generation)

**Computer Vision (CV):** object detection, segmentation, instance segmentation, OpenCV

**Recommender Systems:** collaborative filtering, ranking losses, offline/online evaluation

**Reinforcement Learning (RL):** MDPs, Q-learning, policy gradients

**Projects:** Build a QA pipeline (NLP), or an object detection demo (CV), or a recommender prototype

---

### Phase 9 — Research & Advanced Production
**Skills:**
- Reproduce research papers, distributed training, mixed precision, pruning & quantization
- Cost/accuracy trade-offs and optimizations

**Activities:**
- Reproduce a recent paper and write a technical blog post describing the experiments and results

---

## Portfolio Projects & Deliverables
For every project include:
- `notebooks/` (clean, annotated)
- `src/` (production-ready modules)
- `app/` (FastAPI inference server)
- `Dockerfile` and optional `docker-compose.yml`
- `README.md` with problem, approach, results, and how-to-run
- Short demo GIF or 2–5 minute video

**Project progression:** EDA → Baseline ML → Deep Learning → Serving → MLOps pipeline

---

## Recommended Repo Structure
```
AI_Project/
├─ notebooks/
├─ src/
│  ├─ data/
│  ├─ models/
│  └─ utils/
├─ app/
├─ data/
├─ docs/
├─ Dockerfile
├─ requirements.txt
├─ environment.yml
└─ README.md
```

**Notebook template:** Title → Goal → Requirements → Imports → EDA → Preprocessing → Modeling → Evaluation → Conclusions → Next steps

---

## README & Repository Polishing Tips
- Add badges: build, tests, license, python version
- Quickstart instructions (1-2 commands to run locally)
- Provide example inputs/outputs and a short demo GIF
- Add a LICENSE (MIT suggested) and CONTRIBUTING.md
- Add GitHub Actions to run tests and lint on PRs

**Badges example snippet (place at top of README):**
```
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
```

---

## License & Contribution
**Suggested license:** MIT (add `LICENSE` file)
**Contributing:** Add `CONTRIBUTING.md` with steps to run notebooks, tests, and coding style rules (black, flake8)

---

## Checklist (to make README professional)
- [ ] Clear description & goals
- [ ] Table of contents
- [ ] Quickstart & install instructions
- [ ] env file (`environment.yml`) or `requirements.txt`
- [ ] Demo GIF or video
- [ ] Notebooks with explanations and math
- [ ] `src/` production-ready code
- [ ] Dockerfile and CI workflow
- [ ] LICENSE and CONTRIBUTING

---

---

# النسخة العربية — خارطة تعلم كاملة في الذكاء الاصطناعي وعلوم البيانات (جاهزة كـ README)

> نسخة عربية مفصّلة وواضحة من الوثيقة الإنجليزية أعلاه، تشبهها تمامًا في البنية والمحتوى، بهدف النشر على GitHub كبداية مرجعية مفيدة لجميع المتعلمين.

## جدول المحتويات
1. [هدف المشروع والجمهور](#هدف-المشروع-والجمهور)
2. [كيفية استخدام خارطة التعلم هذه](#كيفية-استخدام-خارطة-التعلم-هذه)
3. [المراحل العامة للتعلم](#المراحل-العامة-للتعلم)
4. [الخطة التفصيلية مرحلة بمرحلة]
5. [مشروعات المحفظة وتسليماتها]
6. [هيكل المشروع المقترح وقوالب الملفات]
7. [نصائح لتجميل README وتهيئة المستودع]
8. [الترخيص والمساهمة]
9. [قائمة التحقق]

---

## هدف المشروع والجمهور
**الهدف:** إعداد خارطة تعلم كاملة تغطي كل ما يحتاجه شخص ليصبح محترفًا في علوم البيانات وهندسة الذكاء الاصطناعي — من إعداد البيئة الرياضية والبرمجية وحتى النشر والإنتاج والبحث.

**الجمهور:** طلاب، متعلمون ذاتيًا، مهندسو ML/AI المبتدئون، والهواة.

---

## كيفية استخدام خارطة التعلم هذه
1. ابدأ بـ **إعداد البيئة** و**أساسيات البرمجة والبيانات**.
2. طبّق الرياضيات عمليًا أثناء المشاريع (PCA، انحدار، الخ).
3. انتقل إلى التعلم الآلي التقليدي، ثم التعلم العميق، ثم هندسة النماذج ثم MLOps.
4. اختَر التخصص الذي يهمك (NLP، CV، recommender، الخ) بعد إتقان الأساسيات.
5. لكل موضوع: تعلّم النظرية، اتّبع درسًا عمليًا، وانفذ مشروعًا قابلًا لإعادة التشغيل مع كود جاهز للإنتاج.

---

## المراحل العامة للتعلم
- المرحلة 0 — البيئة والأساسيات
- المرحلة 1 — البرمجة وأساسيات البيانات
- المرحلة 2 — الرياضيات للـ ML
- المرحلة 3 — التعلم الآلي التقليدي
- المرحلة 4 — التعلم العميق والمعماريات الحديثة
- المرحلة 5 — هندسة الذكاء الاصطناعي ونشر النماذج
- المرحلة 6 — MLOps ودورة الإنتاج
- المرحلة 7 — البيانات الكبيرة والبث
- المرحلة 8 — التخصّصات
- المرحلة 9 — البحث والمواضيع المتقدمة

---

## الخطة التفصيلية مرحلة بمرحلة

### المرحلة 0 — البيئة والأساسيات
**المهارات:**
- إدارة بيئات بايثون (conda/venv)
- أوامر الشل الأساسية، SSH، إدارة الملفات
- Git وGitHub (الفروع، PRs، القضايا)

**الأدوات:** Python 3.9+، Conda/venv، Git، VSCode، JupyterLab

**المخرجات:** `environment.yml` أو `requirements.txt`، README بسيط، أول Commit

---

### المرحلة 1 — البرمجة وأساسيات البيانات
**المهارات:**
- بايثون المتقدم (OOP، الحلقات، الوحدات، virtual envs)
- pandas وNumPy لمعالجة البيانات
- تصور البيانات: matplotlib، seaborn، plotly
- SQL للاستعلام عن البيانات وعمليات الربط

**مشروعات مقترحة:**
- تقرير EDA (مجموعات بيانات عامة مثل Titanic)
- خط أنابيب تنظيف بيانات قابل للإعادة

---

### المرحلة 2 — الرياضيات للـ ML
**الموضوعات:** الجبر الخطي، الاحتمالات والإحصاء، التفاضل والتكامل، أساسيات حساب المصفوفات في backpropagation

**كيفية التطبيق:** تنفيذ PCA وانحدار خطي وGD من الصفر باستخدام NumPy

---

### المرحلة 3 — التعلم الآلي التقليدي
**المهارات:** خوارزميات إشرافية وغير إشرافية، تقييم النماذج، هندسة الميزات

**الأدوات:** scikit-learn، XGBoost، LightGBM

**مشروعات:** مشروع تصنيف جدولي مع تفسير النموذج (SHAP)

---

### المرحلة 4 — التعلم العميق والمعماريات الحديثة
**المهارات:** شبكات التفاف (CNN)، RNN، محولات (Transformers)، fine-tuning

**الأدوات:** PyTorch، TensorFlow/Keras، Hugging Face

**مشروعات:** CNN على CIFAR-10، fine-tune transformer لمهمة نصية

---

### المرحلة 5 — هندسة الذكاء الاصطناعي ونشر النماذج
**المهارات:** تحويل النماذج إلى صيغ إنتاجية، بناء API للخدمة، حاويات Docker

**الأدوات:** FastAPI، Docker، ONNX

**مشروعات:** نشر نموذج عبر FastAPI + Docker

---

### المرحلة 6 — MLOps ودورة الإنتاج
**المهارات:** تتبع التجارب، نسخ البيانات، CI/CD، مراقبة النماذج

**الأدوات:** MLflow، DVC، GitHub Actions، Airflow

**مشروعات:** أنبوب تدريب مع DVC وMLflow

---

### المرحلة 7 — البيانات الكبيرة والبث
**المهارات:** معالجة موزعة، ETL، بث البيانات

**الأدوات:** Spark، Kafka

**مشروعات:** ETL عبر Spark، مستهلك Kafka لتحديث Feature Store

---

### المرحلة 8 — التخصّصات
**NLP:** tokenization، embeddings، transformers، RAG

**CV:** كشف الكائنات segmentation

**Recommender:** أنظمة توصية مُتقدمة

**RL:** Q-learning، policy gradients

---

### المرحلة 9 — البحث والمواضيع المتقدمة
**المهارات:** إعادة تنفيذ الأوراق البحثية، التدريب الموزع، تحسين استخدام الموارد الحاسوبية

**نشاط مُوصى به:** تنفيذ ورقة بحثية حديثة وكتابة تقرير تقني

---

## مشروعات المحفظة وتسليماتها
لكل مشروع:\n- notebooks/ نظيفة ومُعلّقة\n- src/ كود إنتاجي\n- app/ خادم FastAPI\n- Dockerfile\n- README يشرح المشكلة والطريقة والنتائج\n- GIF أو فيديو قصير\n\n---\n\n## هيكل المشروع المقترح\n```\nAI_Project/\n├─ notebooks/\n├─ src/\n├─ app/\n├─ data/\n├─ Dockerfile\n├─ requirements.txt\n└─ README.md\n```\n\n---\n\n## نصائح لتجميل README\n- شارات badges\n- أوامر Quickstart سريعة\n- مثال لإدخال/إخراج\n- ملف LICENSE (MIT مقترح)\n- ملف CONTRIBUTING.md\n\n---\n\n## الترخيص والمساهمة\n**ترخيص مُقترح:** MIT\n**المساهمة:** أضف CONTRIBUTING.md يشرح كيفية تشغيل المستودع والاختبارات\n\n---\n\n## قائمة التحقق\n- [ ] وصف واضح\n- [ ] جدول محتويات\n- [ ] Quickstart\n- [ ] environment.yml أو requirements.txt\n- [ ] Demo GIF\n- [ ] Notebooks مع شرح\n- [ ] Dockerfile\n- [ ] LICENSE\n\n---\n\n*End of document — English followed by Arabic.*

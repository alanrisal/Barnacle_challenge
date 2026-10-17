# Barnacle annotation: working preparation pipeline

Region selection and initial instance-segmentation dataset preparation are implemented.
They run on CPU with OpenCV and NumPy; model training is deferred.

A [Colab T4 notebook](notebooks/barnacle_training_colab.ipynb) and
[training guide](docs/COLAB_TRAINING.md) now provide PointRend setup, a short
diagnostic, and optional fine-tuning. GPU execution is to be verified in Colab.

The [unseen-image demo](docs/UNSEEN_DEMO.md) adds checkpoint export, tiled mask
inference on both unseen photographs, overlap reconciliation and notebook reports.

```powershell
.\.venv\Scripts\python.exe -m barnacle_challenge prepare-data --output artifacts/pretraining
.\.venv\Scripts\python.exe -m barnacle_challenge validate-data artifacts/pretraining
```

Use a new output directory for subsequent builds. The supplied annotations are
provisional and generate review overlays, issue crops, lossless instance labels,
and scene-separated COCO tiles. They are not verified model predictions.

See [the preparation guide](docs/PRETRAINING.md) for region editing with zoom/pan,
artifact descriptions, label review, validation, and the future model data contract.
The original brainstorming notes are preserved below.

# Project - Thought process

## Initial thoughts

We need to deal with a certain image region and compute the number of barnacles in the region. 

Images are huge in size which is good, enables better masking and selection. 

We should think about the different requirements of the user 



User wants fast and reliable image annotation, scientist pipeline means the immage annotation model should be highly accurate but cannot be 100% accurate, so there should be a chance for manual annotation as well. 

We should also present multiple approaches to annotation -- this includes demonstrating three different methods and the predicted counts for each one. 

The overall strategy for this should be building the i







Here is a rough step-by-step process to follow, though you are welcome to deviate as necessary

Brainstorm an idea for a system that can help the scientists get their work done faster. This only needs to be at a level of detail of a few sentences.

Identify one or more critical subtasks that are necessary to solving the task. Define these tasks clearly by thinking about what information each one will receive and what outputs they are responsible for providing.

💡

Note: The next three bullets are written in a modeling/automation focused way. If you want to focus more on visualization, feel free to deviate a bit. There may not be metrics with which to evaluate, for instance. Instead, focus on whether your visualization actually gives you the insights you were hoping to obtain

Pick a subtask that you are interested in working on, and think about how you can evaluate performance on it. What metrics do you care about, and is it practical for you to compute them with just the data we’ve provided for you?

Build a prototype which attempts to do your identified subtask

Analyze the performance of your prototype, and report your results!

💡

If you’re prototype doesn’t perform that well, you do not have to go back and make another one! We just want an honest evaluation

Make some conclusions! If you built some type of automation prototype, is your approach worth pursuing? What might work better? If you build a visualization, what does your prototype tell you?

Tell us about your learning process. As stated, one of the main points of this application process is for you to learn new things. What is something you learned that excited you?









Model-Focused
Here are places you could start if you want to focus on the actual automation of the process. These are roughly ordered by increasing levels of complexity.

While this is a list of tools you could try, you also need to clearly define the inputs and output formats of your model. There is not one obvious way to systematize this problem and we want to see what you come up with. Just as important is coming up with good evaluation metrics that show whether your proof of concept has value.

💡

Note: We are generally not more impressed by submissions which use fancier tools. While the added complexity required to use a fancy tool well can add a little bit of value to your application, it is much much better to use a simpler and more understandable tool in a more interesting way.

Traditional Computer Vision/Image Processing

OpenCV. Play around with kernels and see if you can make one that helps you find barnacles. Even more simply, look for circles or ellipses in the image.

Machine Learning Approaches

Learn about what Scikit-Learn has to offer. Perhaps you could do something with a classification algorithm?

Deep Learning Approaches

Work with CNNs in PyTorch and see if you can train one to identify barnacles

State of the Art (SOTA) Deep Learning (or at least close to)

Mask R-CNN

Segment Anything

If you are interested in presenting your results in an interactive way, consider building a streamlit demo to showcase your prototype!

Visualization-Focused
Here are places you could start if you want to focus on how to leverage visualizations to improve the process. If you want to go down this route, feel free to assume that some sort of model exists without creating it. If you do that, just clearly define what the input and output of the model is and how it would be used, and come up with some mock version that you can use in your visualizations.

Exploring the data

Looking at the mask files could yield some interesting results. Can you compute metrics on the average size of the barnacles, their spacing, etc, that might give the scientists and model developers new insights?

Human-Model interaction

Prototype some feature of an interface that allows humans to interact with a model (correcting its mistakes, double checking its results, etc)

Frameworks/Tools:

Python visualization libraries

Most are based on matplotlib, but some common high level interfaces are seaborn, plotly, and plotnine

Streamlit is not a visualization library per se, but is a great tool for adding interactivity to your visuals and packaging them up in a small prototype/demo app

Javascript visualization libraries

More common for more interactive visualizations/visualizations which will get embedded in a web page

D3.js is the main library that DALI uses

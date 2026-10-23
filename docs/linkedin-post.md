🪄 Build an AR effect on a small part of an image—and let Landpy bring the edit back home.

That is the idea behind Landpy, my Python package for small augmented-reality experiments: crop a face, use landmarks to select a region, add a color effect or transparent overlay, and return the result to the original image.

⏪ One of its core ideas is retrace(): record a crop, edit it, then retrace it to place the edited pixels back on the original canvas. It keeps the edit. Checkpoint undo is there when you want the original state back.

I started this project years ago and never found the time to finish it. With AI’s help, I finally could. Thank you, ChatGPT, for helping me complete the code, tests, documentation, and tutorial journey. 🤖❤️

📓 The eight notebooks start with a preview of a face effect, then build the pieces step by step. You can see masks, landmarks, and debugger snapshots along the way—and finish with the full crop → edit → retrace workflow.

It is a small toolkit for experimenting with image-based AR effects, with runnable examples and room to try your own ideas. I would love to see what you make with it. ✨

🚀 Start here (Colab): https://colab.research.google.com/github/bateni1380/landpy/blob/master/tutorials/1-images-and-coordinates.ipynb
📚 All tutorials: https://github.com/bateni1380/landpy/tree/master/tutorials
💻 GitHub: https://github.com/bateni1380/landpy
📦 Install: pip install landpy
https://pypi.org/project/landpy/

#Python #OpenSource #AugmentedReality #ComputerVision #ImageProcessing #AI

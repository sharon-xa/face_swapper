# Run the project

1. Install prerequisites:
	- Python.
	- Desktop Development with C++ (via Visual Studio Installer).
2. Make sure that you have 2 images:
	- source.jpg
	- target.jpg
3. Make sure to have the inswapper model inside the "models" folder.
4. Create a virtual environment `python -m venv .venv`.
5. Activate the env `.\.venv\Scripts\activate`.
6. Install all dependencies (Ex: pip install opencv-python):
	- opencv-python
	- insightface
	- matplotlib
	- onnxruntime
7. Run the program `python main.py`.
8. For targeting one face only `python main.py 1`.
9. You can write 1 or more, but not more than the faces you have in the target image.

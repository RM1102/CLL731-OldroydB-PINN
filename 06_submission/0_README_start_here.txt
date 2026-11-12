CLL731 TERM PAPER (ML)
Rahul Arvind Masand (2022CH71476), IIT Delhi, November 2026

Pressure-driven flow of the viscoelastic Oldroyd-B fluid in narrow non-uniform
geometries (Boyko & Stone 2022, J. Fluid Mech. 936, A23), re-solved with a
physics-informed neural network (PINN).

Start with the 2-page summary. Everything else is optional.


1. WHAT IS IN THIS FOLDER
-------------------------
  1_Summary_2_pages.pdf    What the project is about, the papers used, what was
                           done, the results and why it is useful. Read first.

  2_Presentation.pptx      The full story in 20 slides, with every figure and
                           the method in detail.

  3_Full_report.html       The complete report with all the code and charts.
                           Double-click to open it in any web browser.
                           Nothing to install.

  0_README_start_here.txt  This file: where the code is, and how to run it on
                           your own computer.


2. WHERE THE CODE IS
--------------------
  All the code, the 17 trained networks and the research notes are on GitHub:

      https://github.com/RM1102/CLL731-OldroydB-PINN

  The GitHub page opens with a short guide. You can also read the full report
  there without downloading anything: click the folder "05_notebook", then
  the file "CLL731_OldroydB_PINN.ipynb".


3. RUN IT ON YOUR OWN COMPUTER (Windows or Mac, no GPU needed)
--------------------------------------------------------------
  Step 1. Install Python (skip this if Python 3.10 or newer is installed)
    - Go to https://www.python.org/downloads/ and click the yellow
      "Download Python" button.
    - Windows: open the downloaded file. On the FIRST screen, tick the box
      "Add python.exe to PATH" at the bottom, then click "Install Now".
    - Mac: open the downloaded file and click "Continue" until it finishes.

  Step 2. Download the code from GitHub
    - Open https://github.com/RM1102/CLL731-OldroydB-PINN
    - Click the green "<> Code" button above the list of files.
    - In the menu that opens, click "Download ZIP" (the last line).
    - Unzip the downloaded file "CLL731-OldroydB-PINN-main.zip":
        Windows: right-click it, choose "Extract All...", then "Extract".
        Mac: double-click it.

  Step 3. Start it
    Windows:
    - Open the unzipped folder and double-click "setup_and_run.bat".
    - If a blue box says "Windows protected your PC", click "More info",
      then "Run anyway". Windows shows this for any script downloaded from
      the internet.
    Mac:
    - Open the unzipped folder. Right-click (or Control-click)
      "setup_and_run_mac.command", choose "Open", then click "Open" again.
      A Mac blocks a plain double-click on downloaded scripts the first time.
    - If that does not work: open the Terminal app, type  bash  and a space,
      drag the file "setup_and_run.sh" from the folder into the Terminal
      window, and press Enter.

    The first time, a window shows the setup. It downloads about 1 GB
    (PyTorch and Jupyter) and takes 5 to 15 minutes. After that it starts in
    seconds. Everything is installed inside the folder itself (in ".venv");
    nothing else on the computer is changed.

  Step 4. Choose from the menu (press the number; on a Mac, then press Enter)
    1  PINN explorer window: pick a channel and a field, move the De slider,
       and type a point (Z, Y) to read the predicted values there.
    2  The report notebook, in your browser. Choose "Run" > "Run All Cells";
       it takes about 20 seconds.
    3  Quick check that everything works (about 10 seconds).
    4  Retrain every network from scratch.
       WARNING: this takes about 4 HOURS on a laptop, and it is not needed,
       because the trained networks are already included.


4. USING THE TRAINED NETWORK FOR PREDICTIONS
--------------------------------------------
  A trained PINN works like a formula: give it a position in the channel and
  a Deborah number De, and it returns the velocity, pressure and polymer
  stress there in a fraction of a second. Nothing is solved again.

  - Easiest: menu option 1, the explorer window.
  - From the command line, inside the unzipped folder:
        Windows:  .venv\Scripts\python predict.py --case eps0.02 --De 0.2 --dp
        Mac:      .venv/bin/python predict.py --case eps0.02 --De 0.2 --dp
    This prints the pressure drop, also in pascals for the paper's channel.
    Other options:  --list (what is available),  --point 0.5 0.3 (every value
    at Z = 0.5, Y = 0.3),  --csv fields.csv (a whole field as a table).

  The networks are trained for the paper's channel (4:1 contraction, polymer
  share of the viscosity 0.4) and De from 0 to 0.3. Outside that range their
  answers are not reliable.


5. IF SOMETHING GOES WRONG
--------------------------
  - "Python was not found": do step 1, and on Windows make sure the box
    "Add python.exe to PATH" was ticked. Then run the script again.
  - The setup stopped while downloading: check the internet connection and
    run the script again. It carries on from where it stopped.
  - To remove everything: delete the unzipped folder.


Use of AI: the literature search, code and text of this project were developed
with the help of an AI assistant (Claude), as stated in the project proposal.
All results were checked against the paper and an independent solver.

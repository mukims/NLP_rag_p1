#!/usr/bin/env python3
"""
Multimodal RAG Agent - Orchestrator
=====================================
This agent guides you through the full RAG pipeline:
  1. Asks for the location(s) of your PDF files.
  2. Copies them into the `raw/` directory.
  3. Runs image/figure extraction (extract_images.py).
  4. Builds the vector + keyword search index (build_index.py).
  5. Optionally launches the Streamlit query UI (app.py).

Usage:
    python agent.py
"""

import os
import sys
import shutil
import glob
import subprocess
import textwrap

# -----------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------

def header(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def step(n: int, text: str):
    print(f"\n[Step {n}] {text}")


def success(msg: str):
    print(f"  ✅  {msg}")


def warn(msg: str):
    print(f"  ⚠️   {msg}")


def error(msg: str):
    print(f"  ❌  {msg}")


def divider():
    print("  " + "-" * 56)


# -----------------------------------------------------------------------
# Change working directory to the script's own directory so that all
# relative paths (raw/, extracted_data/, index/) resolve correctly.
# -----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
EXTRACTED_DIR = os.path.join(SCRIPT_DIR, "extracted_data")
INDEX_DIR = os.path.join(SCRIPT_DIR, "index")


# -----------------------------------------------------------------------
# Step 1 – Ask for PDF locations
# -----------------------------------------------------------------------

def collect_pdf_paths() -> list[str]:
    """
    Interactively ask the user for PDF file or directory paths.
    Returns the final resolved list of .pdf file paths.
    """
    header("Step 1 · Locate Your PDFs")
    print(textwrap.dedent("""
        You can provide:
          • The path to a single PDF file, e.g.  /Users/you/paper.pdf
          • The path to a directory that contains PDFs, e.g.  /Users/you/docs/
          • Multiple paths separated by commas

        Type 'done' on an empty line when finished.
    """))

    collected: list[str] = []

    while True:
        raw = input("  PDF path(s): ").strip()

        if raw.lower() in ("done", ""):
            if collected:
                break
            warn("No paths entered yet. Please provide at least one PDF.")
            continue

        # Support comma-separated input
        entries = [e.strip().strip('"').strip("'") for e in raw.split(",")]

        for entry in entries:
            entry = os.path.expanduser(entry)  # handle ~/... paths

            if os.path.isfile(entry) and entry.lower().endswith(".pdf"):
                collected.append(entry)
                success(f"Found PDF: {entry}")

            elif os.path.isdir(entry):
                pdfs = glob.glob(os.path.join(entry, "**", "*.pdf"), recursive=True)
                if pdfs:
                    collected.extend(pdfs)
                    success(f"Found {len(pdfs)} PDF(s) in: {entry}")
                else:
                    warn(f"No PDF files found in directory: {entry}")

            else:
                error(f"Not a valid .pdf file or directory: {entry}")

        divider()
        print(f"  📄  Total PDFs collected so far: {len(collected)}")

        more = input("  Add more paths? [y/N]: ").strip().lower()
        if more not in ("y", "yes"):
            break

    if not collected:
        error("No valid PDF files provided. Exiting.")
        sys.exit(1)

    return collected


# -----------------------------------------------------------------------
# Step 2 – Copy PDFs into raw/ directory
# -----------------------------------------------------------------------

def stage_pdfs(pdf_paths: list[str]) -> int:
    """Copy the user's PDFs into raw/ and return the count staged."""
    header("Step 2 · Staging PDFs → raw/")
    os.makedirs(RAW_DIR, exist_ok=True)

    staged = 0
    for src in pdf_paths:
        dest_name = os.path.basename(src)
        dest = os.path.join(RAW_DIR, dest_name)

        if os.path.abspath(src) == os.path.abspath(dest):
            success(f"Already in place: {dest_name}")
            staged += 1
            continue

        try:
            shutil.copy2(src, dest)
            success(f"Copied: {dest_name}")
            staged += 1
        except Exception as exc:
            error(f"Could not copy {src}: {exc}")

    print(f"\n  Staged {staged}/{len(pdf_paths)} PDF(s) into '{RAW_DIR}'.")
    return staged


# -----------------------------------------------------------------------
# Step 3 – Run extraction pipeline
# -----------------------------------------------------------------------

def run_extraction():
    """Import and run extract_images.main() in-process."""
    header("Step 3 · Extracting Figures & Tables from PDFs")
    print("  This may take a while depending on the number of pages …\n")

    try:
        from extract_images import main as extract_main
        extract_main(raw_dir=RAW_DIR)
        success("Extraction complete.")
    except FileNotFoundError as exc:
        error(str(exc))
        print(textwrap.dedent("""
            Please download the LayoutParser / Detectron2 model weights:
              wget -O model_final.pth \\
                https://dl.fbaipublicfiles.com/detectron2/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth
        """))
        sys.exit(1)
    except ImportError as exc:
        error(f"Import error: {exc}")
        print("  Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as exc:
        error(f"Extraction failed: {exc}")
        raise


# -----------------------------------------------------------------------
# Step 4 – Build index
# -----------------------------------------------------------------------

def run_indexing():
    """Import and run build_index.build_index() in-process."""
    header("Step 4 · Building Search Index (CLIP + SentenceTransformer + BM25)")
    print("  This embeds extracted figures and captions …\n")

    try:
        from build_index import build_index
        build_index()
        success("Indexing complete.")
    except ImportError as exc:
        error(f"Import error: {exc}")
        print("  Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as exc:
        error(f"Indexing failed: {exc}")
        raise


# -----------------------------------------------------------------------
# Step 5 – Optionally launch the Streamlit app
# -----------------------------------------------------------------------

def prompt_launch_ui():
    """Ask the user whether to launch the Streamlit query interface."""
    header("Step 5 · RAG System Ready 🎉")
    success("Your index has been built and is ready to query.")
    print(f"\n  Index location : {INDEX_DIR}")
    print(f"  Extracted data : {EXTRACTED_DIR}")
    print()

    launch = input("  Launch the Streamlit UI now? [Y/n]: ").strip().lower()
    if launch in ("", "y", "yes"):
        print("\n  Starting Streamlit … (press Ctrl+C to stop)\n")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", "app.py"],
                cwd=SCRIPT_DIR,
                check=True
            )
        except KeyboardInterrupt:
            print("\n  Streamlit stopped by user.")
        except FileNotFoundError:
            error("streamlit not found. Run:  pip install streamlit")
        except subprocess.CalledProcessError as exc:
            error(f"Streamlit exited with code {exc.returncode}.")
    else:
        print("\n  To query manually, run:")
        print("    streamlit run app.py")


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def main():
    header("Multimodal RAG Agent")
    print("  This agent will guide you through building the RAG pipeline.")

    # 1. Collect PDF locations from the user
    pdf_paths = collect_pdf_paths()

    # 2. Stage them into raw/
    staged = stage_pdfs(pdf_paths)
    if staged == 0:
        error("No PDFs were staged. Exiting.")
        sys.exit(1)

    # 3. Extract figures and tables
    run_extraction()

    # 4. Build the search indices
    run_indexing()

    # 5. Optionally launch the UI
    prompt_launch_ui()


if __name__ == "__main__":
    main()

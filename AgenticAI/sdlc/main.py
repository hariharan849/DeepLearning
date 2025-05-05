import sys
sys.path.append('./')

from ui import main_interface

def main():
    app = main_interface.SDLCInterface()
    app.load_streamlit_ui()

if __name__ == '__main__':
    main()
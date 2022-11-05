import os
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

images = os.listdir(Path("generated_samples"))
images = [i for i in images if i.endswith(".png")]

attributes = [i.removesuffix(".png") for i in images]
attributes = [i.split("-") for i in attributes]  # outer radius / thickness / radian

outer_radius = sorted(list(set([float(i[0]) for i in attributes])))
thickness = sorted(list(set([float(i[1]) for i in attributes])))
radian = sorted(list(set([float(i[2]) for i in attributes])))


class Demo:
    def __init__(self):
        self.sidebar()
        self.remove_logo()
        self.main()

    def main(self):
        display = Image.open(f"generated_samples/{self.selected_outer_radius}-{self.selected_thickness}-{self.selected_radian}.png")
        st.image(display, width=500)

    def sidebar(self):
        with st.sidebar:
            st.title("Jastrow Demo")
            self.selected_thickness = st.select_slider(label="Thickness", options=thickness, value=thickness[0])
            self.selected_radian = st.select_slider(label="Radian", options=radian, value=radian[0])
            self.selected_outer_radius = st.selected_slider(label="Radius", options=outer_radius, value=outer_radius[0])

    def remove_logo(self):
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    Demo()

import os
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

images = os.listdir(Path("samples"))
images = [i for i in images if i.endswith(".png")]

attributes = [i.removesuffix(".png") for i in images]
attributes = [i.split("-") for i in attributes]  # outer radius / thickness / radian / distance

attribute_names = ["outer_radius", "thickness", "radian", "distance"]


class Demo:
    def __init__(self):
        # self.outer_radius = sorted(list(set([float(i[0]) for i in attributes]))) # 3800.0
        self.thickness = sorted(list(set([float(i[1]) for i in attributes])))
        self.radian = sorted(list(set([float(i[2]) for i in attributes])))
        self.distance = sorted(list(set([float(i[3]) for i in attributes])))

        self.remove_logo()
        self.sidebar()
        self.main()

    def main(self):
        display = Image.open(Path("samples", f"3600.0-{self.selected_thickness}-{self.selected_radian}-{self.selected_distance}.png"))
        st.image(display, width=display.size[0] // 3)

    def sidebar(self):
        with st.sidebar:
            st.title("Jastrow Demo")

            self.selected_thickness = st.select_slider(label="Thickness", options=self.thickness, value=self.thickness[0])
            self.selected_radian = st.select_slider(label="Curvature", options=self.radian, value=self.radian[0])
            self.selected_distance = st.select_slider(label="Distance", options=self.distance, value=self.distance[0])

    def remove_logo(self):
        hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        # MainMenu {visibility: hidden;}


if __name__ == "__main__":
    Demo()

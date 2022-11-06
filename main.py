import os
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

images = os.listdir(Path("generated_samples"))
images = [i for i in images if i.endswith(".png")]

attributes = [i.removesuffix(".png") for i in images]
attributes = [i.split("-") for i in attributes]  # outer radius / thickness / radian

attribute_names = ["outer_radius", "thickness", "radian"]


class Demo:
    def __init__(self):
        self.outer_radius = sorted(list(set([float(i[0]) for i in attributes])))
        self.thickness = sorted(list(set([float(i[1]) for i in attributes])))
        self.radian = sorted(list(set([float(i[2]) for i in attributes])))

        self.sidebar()
        self.remove_logo()
        self.main()

    def main(self):
        if self.selected_presentation == "Batch":
            display = Image.open(Path(f"grid_{self.selected_category}", f"{self.selected_category_value}.png"))
            st.image(display, use_column_width=True)
        else:
            display = Image.open(
                Path(
                    "generated_samples",
                    f"{self.selected_outer_radius}-{self.selected_thickness}-{self.selected_radian}.png",
                )
            )
            st.image(display, width=500)

    def sidebar(self):
        with st.sidebar:
            st.title("Jastrow Demo")

            self.selected_presentation = st.radio(label="How", options=["Image", "Batch"])

            if self.selected_presentation == "Batch":
                self.selected_category = st.selectbox(label="By", options=attribute_names)
                self.selected_category_value = st.select_slider(
                    label="Value",
                    options=getattr(self, self.selected_category),
                    value=getattr(self, self.selected_category)[0],
                )
            else:
                self.selected_thickness = st.select_slider(
                    label="Thickness", options=self.thickness, value=self.thickness[0]
                )
                self.selected_radian = st.select_slider(label="Radian", options=self.radian, value=self.radian[0])
                self.selected_outer_radius = st.select_slider(
                    label="Radius", options=self.outer_radius, value=self.outer_radius[0]
                )

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

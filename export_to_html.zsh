#!/bin/zsh 
echo "converting notebook into html..."
jupyter-nbconvert --to html_exporter.exporter.CH_TOC growth_application/growth-application.ipynb --output ../../docs/growth-application.html
echo "done!"
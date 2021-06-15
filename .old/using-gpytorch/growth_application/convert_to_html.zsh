#!/bin/zsh 
echo "converting notebook into html..."
jupyter-nbconvert --to exporter.exporter.CH_TOC growth-application.ipynb --output ../../docs/growth-application.html
echo "copying images..."
cp -r img docs
echo "done!"
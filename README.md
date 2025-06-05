# A fast and interpretable prediction system for the Site Of Origin (SOO) of Outflow Tract Ventricular Arrhythmias (OTVAs)

## Introduction

This repository contains the development of a system of machine learning models for predicting the Site Of Origin (SOO) of Outflow Tract Ventricular Arrhythmias (OTVAs), from patient cases that consist mainly of ECGs and demographic data.

OTVAs are premature ventricular beats that can lead to significant morbidity and mortality if not properly diagnosed and treated. Accurate localization of the arrhythmia's origin is crucial for effective treatment, including catheter ablation procedures, and detecting the arrhythmia's origin from purely eye-inspecting ECG data can be a challenging task.

## Project tasks

This project is divided into two main tasks, each focusing on a different aspect of the arrhythmia classification problem:

1. **Part 1: Left vs. Right Outflow Tract**  
   Classification of arrhythmia as originating from either the Left Ventricular Outflow Tract (LVOT) or the Right Ventricular Outflow Tract (RVOT).

2. **Part 2: Sub-regional localization**  
   Discrimination between origins at the Right Coronary Cusp (RCC) and the aortomitral commissure (the mitral-aortic continuity).

## Goals

The goal of this project is to develop a system of models for each task that can **accurately** predict the Site of Origin of OTVAs. Given the medical context, the system should also be **interpretable**, providing an explanation into the *learned* decision-making process of the models. Additionally, the system should have **fast and lightweight inference**, enough for real-time clinical applications running on edge devices.

## About the process

The project is designed to be **easily reproducible**, with all the code and data organized in a clear and structured manner. This is why the whole end-to-end pipeline is contained in a single Jupyter Notebook ([project_arrhythmias.ipynb](project_arrhythmias.ipynb)), which includes all the steps from data preprocessing to model evaluation. This notebook explains our approach in detail, and can be executed to reproduce the results on any machine.

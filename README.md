# 🎓 Master's Independent Study — LSCSE

[![Excel Simulation](https://img.shields.io/badge/Tool-Excel-green?logo=microsoft-excel)](#)

---

<div align="center">
  <a href="https://www.siit.tu.ac.th/">
    <img src="https://admissions.siit.tu.ac.th/wp-content/uploads/2023/06/cropped-TU-SIIT1992-01.png" alt="SIIT Logo" width="150"/>
  </a>
  <h3>Master’s Independent Study — LSCSE</h3>
  <p><b>Sirindhorn International Institute of Technology (SIIT), Thammasat University</b></p>
  <p><b>Advisor:</b> Dr. Jirachai Buddhakulsomsiri</p>
</div>

---

## 📋 Project Title

> Simulation-Based Inventory Optimization for Toy Retailers Using Base-Stock Policy

---

## 🎯 Objectives

- Minimize total inventory costs (holding, shortage, ordering).
- Maintain high service levels via optimal base-stock levels.
- Provide a low-cost, spreadsheet-based decision-support tool for toy retailers.

---

## ❗ Problem Statement

Small-to-mid-sized toy retailers often suffer from:
- High demand uncertainty and short product life cycles.
- Seasonal demand patterns.
- Limited access to advanced inventory systems.

This results in stockouts or excessive inventory. Our study creates a simulation-based tool to help such businesses make better inventory decisions without complex software.

---

## 🧠 Methodology

We simulate a **base-stock inventory policy** under a **lost sales environment** (fixed 2-day lead time).

**Key Steps:**
1. Aggregate historical toy demand from a Kaggle dataset.
2. Estimate the empirical demand distribution during lead time (DDLT).
3. Calculate the expected shortage $E(S)$.
4. Simulate total costs across various base-stock levels.
5. Optimize using an iteration-based search to find the best policy.

---

## 🧮 Cost Model

The total inventory cost is modeled as the sum of three components:

$$
\text{Total Cost} = \text{Ordering Cost} + \text{Holding Cost} + \text{Shortage Cost}
$$

- **Ordering Cost:** A fixed cost incurred per ordering cycle.
- **Holding Cost:** The cost of carrying inventory, calculated as $CH \cdot \frac{S}{2}$.
- **Shortage Cost:** The cost of lost sales, calculated as $CS \cdot E(S)$.

---

## 📊 Tools & Technologies

- **Excel / Google Sheets** for simulation and modeling.
- **Kaggle Retail Forecasting Dataset** for historical demand data.
- Manual modeling and iterative calculations (no programming required for the core tool).

---

## 📈 Key Results

- Identified an optimal base-stock level that minimizes total costs while balancing the service level.
- Developed a **dynamic, spreadsheet-based inventory decision-support tool**.
- The method is highly suitable for retailers without access to expensive ERP or advanced inventory management systems.

---

## 💻 Code & Simulation

You can explore the simulation logic and interact with the model directly via the Google Colab notebook, which replicates the spreadsheet logic in Python for demonstration.

<a href="https://colab.research.google.com/drive/1ZKxxbaGxrzy3-DLhAEYlIKTKCAnKtrf1?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## 📁 Deliverables

| Final Report | Simulation Workbook | Presentation Slides |
| :----------: | :-------------------: | :-------------------: |
| **[View on Google Docs](https://docs.google.com/document/d/1Iq0yay1xUAMxYXiJTPcNNm_MyfQfmtbnR2q3RRzasLI/edit?usp=sharing)**<br><br><a href="https://docs.google.com/document/d/1Iq0yay1xUAMxYXiJTPcNNm_MyfQfmtbnR2q3RRzasLI/edit?usp=sharing"><img src="https://img.icons8.com/color/96/google-docs--v1.png" width="64" alt="Google Docs Icon"/></a> | **[View on Google Sheets](https://docs.google.com/spreadsheets/d/1xkvY5pgZ9h3RBFdfx8_XHcTr8V5JO7TQA7KUig3dHlQ/edit?usp=sharing)**<br><br><a href="https://docs.google.com/spreadsheets/d/1xkvY5pgZ9h3RBFdfx8_XHcTr8V5JO7TQA7KUig3dHlQ/edit?usp=sharing"><img src="https://img.icons8.com/color/96/google-sheets.png" width="64" alt="Google Sheets Icon"/></a> | **[View on Google Slides](https://docs.google.com/presentation/d/1Y0cwvvE2SQW9rWCOFuLykTLfr5jwZksDlXM--qWKv28/edit?usp=sharing)**<br><br><a href="https://docs.google.com/presentation/d/1Y0cwvvE2SQW9rWCOFuLykTLfr5jwZksDlXM--qWKv28/edit?usp=sharing"><img src="https://img.icons8.com/color/96/google-slides.png" width="64" alt="Google Slides Icon"/></a> |

---

## 👨‍🏫 Advisor

<div align="center">
  <img src="https://lh6.googleusercontent.com/uRbvaXs8JX4BmFXOXmr_11oAZO2PPEWxi7Epn-ZaMYWlxhtZxIZoqXwEvmw-7aPhZ9c7LmQyz7lrwI_vaqKlWH2yc_lT_Hn_sB4vr370UNLGHr8Eor5OGgBWUhpCDrdJZvM3CZUlPNgeosiG8V1Be4ee5rF9776gfjj7l8-xjl1e1lFWHRd-lQ=w1280" alt="Dr. Jirachai Buddhakulsomsiri" width="160" style="border-radius: 50%;"/>
  <br>
  <b>Dr. Jirachai Buddhakulsomsiri</b>
  <p>Associate Professor<br>Sirindhorn International Institute of Technology (SIIT)</p>
</div>

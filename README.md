# 🎓 Master's Independent Study — LSCSE

[![Excel Simulation](https://img.shields.io/badge/Tool-Excel-green?logo=microsoft-excel)](#)

---

<div align="center">
  <img src="https://admissions.siit.tu.ac.th/wp-content/uploads/2023/06/cropped-TU-SIIT1992-01.png" height="100"/>
  <br/><br/>
  <h3>Master’s Independent Study — LSCSE</h3>
  <b>Sirindhorn International Institute of Technology (SIIT), Thammasat University</b>  
  <br/>
  <b>Advisor:</b> Dr. Jirachai Buddhakulsomsiri  
</div>

---

## 📋 Project Title

**Simulation-Based Inventory Optimization for Toy Retailers Using Base-Stock Policy**

---

## 🎯 Objectives

- Minimize total inventory costs (holding, shortage, ordering)
- Maintain high service levels via optimal base-stock levels
- Provide a low-cost, spreadsheet-based decision-support tool for toy retailers

---

## ❗ Problem Statement

Small-to-mid-sized toy retailers often suffer from:
- High demand uncertainty and short product life cycles  
- Seasonal demand patterns  
- Limited access to advanced inventory systems  

This results in stockouts or excessive inventory. Our study creates a simulation-based tool to help such businesses make better inventory decisions without complex software.

---

## 🧠 Methodology

We simulate a **base-stock inventory policy** under a **lost sales environment** (fixed 2-day lead time).

**Key Steps:**
1. Aggregate historical toy demand from Kaggle dataset  
2. Estimate empirical demand distribution during lead time (DDLT)  
3. Calculate expected shortage \( E(S) \)  
4. Simulate total costs across base-stock levels  
5. Optimize using an iteration-based search (manual EOQ + CSL + E(S))

---

## 🧮 Cost Model

\[
\text{Total Cost} = \text{Ordering Cost} + \text{Holding Cost} + \text{Shortage Cost}
\]

- **Ordering Cost:** Fixed per cycle  
- **Holding Cost:** \( CH \cdot \frac{S}{2} \)  
- **Shortage Cost:** \( CS \cdot E(S) \)

---

## 📊 Tools & Technologies

- **Excel** for simulation  
- **Kaggle Retail Forecasting Dataset** for historical demand  
- Manual modeling (no programming required)

---

## 📈 Key Results

- Optimal base-stock level minimizes cost and balances service level  
- Method provides **dynamic, spreadsheet-based inventory decision support**  
- Suitable for retailers without ERP or advanced systems

---

## 📁 Deliverables

- 📊 Excel Simulation Workbook  
- 📽 PowerPoint Presentation ([View Slides](#))  
- 📘 Final Report (optional)

---

## 👨‍🏫 Advisor

<div align="center">
  <img src="images/webpage/JirachaiBuddhakulsomsiri.jpg" height="160"/><br/>
  <b>Dr. Jirachai Buddhakulsomsiri</b><br/>
  Associate Professor<br/>
  Sirindhorn International Institute of Technology (SIIT)
</div>

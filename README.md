# Iris Classification

<p align="center">
<img width="960" height="480" alt="iris flower" src="https://github.com/user-attachments/assets/176dd205-c8b5-414c-893c-261007662570" />
</p>

## Introduction
The UCI Machine Learning Repository [1] maintains 678 datasets collated from differing sources, individually built using differing parameters and sometimes thousands of sample points. Within this collection is the Iris dataset constructed by Robert A. Fisher in 1936, which is one of the earliest known datasets used for classification of Iris flowers using Multivariate Analysis and only contains 150 samples [2]. This dataset allows exploration into machine learning methodology as an educational tool, since the small number of sample points will allow easier manual checking to ensure designed algorithms are performing correctly.

## Methodology
Robert A. Fisher opened many new areas in Multivariate Analysis, one of which is his research on discriminate analysis [3]. Discriminant function analysis is the statistical analysis used to analyse data when the dependent variable or outcome is categorical and independent variable or predictor variable is parametric [4]. In short, if the dependent variable is a dichotomy, there is one discriminant function; if there are k levels of the dependent variable, up to k – 1 discriminant functions can be extracted, and the useful projections can be retained [5]. 

This approach to classification fits very well into the Decision Tree Classifier model, which analyses the entire dataset at the root node, and repeatedly splits the data into smaller subsets based on the most informative features, until a stopping criterion is met [6]. It can also model complex, non-linear relationships between predictor variables and class variables, and they are non-parametric, meaning data does not require standardised or normalised distributions [7].

## Implementation
The code used for investigation has been developed in an intentionally naive way for use of reading by people unfamiliar with Machine Learning code, or code in general. This aids in understanding, appropriation, and
troubleshooting for future improvement.

#### My results, discussions, conclusions, future work, and refences can be found here: [KateMcAlpine_IrisClassificationReport.pdf](https://github.com/katemcalpine/Iris-Classification/blob/main/KateMcAlpine_IrisClassificationReport.pdf)

NB: Some of the information in the reference links are incorrect or outdated. Sourced fixes were found to address the theoretical and programming issues, bringing the finished project up to date with correct information and current components. 

Specifications used for the project are outlined below:

<p align="center">
<table align="center">
<!--<td width="50%" align="center">-->
  <tr>
    <td><b>Language</b></th>
    <td>Python 3.13</th>
  </tr>
  <tr>
    <td><b>Platform</b></th>
    <td>PyCharm Community Edition 2024.3.1.1</th>
  </tr>
  <tr>
    <td><b>Dataset</b></th>
    <td>Iris</th>
  </tr>
  <tr>
    <td><b>Model</b></th>
    <td>Decision Tree Classifier</th>
  </tr>
  <tr>
    <td><b>Libraries</b></th>
    <td>
      <ul>
        <li>Pandas 2.2.3</li>
        <li>numpy 2.2.6</li>
        <li>matplotlib 3.10.3</li>
        <li>seaborn 0.13.2</li>
        <li>os</li>
        <li>graphviz 0.20.3</li>
        <li>scikit-learn 1.6.1</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><b>Code</b></th>
    <td>Found in the <a href="https://github.com/katemcalpine/Iris-Classification">project Github</a>
  </tr>
<!--</td>-->
</table>
</p>

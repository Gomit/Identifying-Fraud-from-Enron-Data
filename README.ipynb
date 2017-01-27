{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### By Meron Goitom\n",
    "\n",
    "# Identifying Fraud from Enron Emails and Financial Data\n",
    "\n",
    "# Project Overview\n",
    "\n",
    "In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.\n",
    "\n",
    "In this project I will build a person of interest (POI) identifier using machine learning, based on financial and email data made public as a result of the Enron scandal. A person of interest (POI) are individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. \n",
    "\n",
    "There are seven major steps in my project:\n",
    "\n",
    "- Data Exploration\n",
    "- Outlier Investigation\n",
    "- Create new features\n",
    "- Features select\n",
    "- Properly scale features\n",
    "- Pick an algorithm\n",
    "- Tune the algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import pickle\n",
    "import sys\n",
    "\n",
    "sys.path.append(\"/Users/merongoitom/Desktop/Nanodegree/ML/Udacity_course/Lesson1_Project/ud120-projects2/tools/\")\n",
    "data_dict = pickle.load(open('/Users/merongoitom/Desktop/Nanodegree/ML/Udacity_course/Lesson1_Project/ud120-projects2/final_project/final_project_dataset.pkl', \"r\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration\n",
    "- Total number of data points\n",
    "- Allocation across classes (POI/non-POI)\n",
    "- Number of features\n",
    "- Are there features with many missing values? etc."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of people in the dataset: 146\n",
      "Number of feature for each person: 21\n",
      "number of POI's in the dataset: 18\n",
      "Total value of the stock - James Prentice: 1095040\n",
      "Nr of email messages to poi - Wesley Colwell: 11\n",
      "Value of stock options - Jeffrey K Skillin: 19250000\n",
      "Total Payments - Lay: 103559793\n",
      "Total Payments - Skilling: 8682716\n",
      "Total Payments - Fastow: 2424083\n",
      "Quantified salary: 95\n",
      "Known email address: 111\n",
      "Nr of 'NaN' for their total payments: 21\n",
      "Percentage of people: 14.3835616438\n",
      "Nr of 'NaN' for their total payments: 0\n",
      "Percentage of people: 0.0\n"
     ]
    }
   ],
   "source": [
    "import Data_Exploration\n",
    "Data_Exploration.Q_A(data_dict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "### Understanding the Dataset\n",
    "\n",
    "POIs are defined as peopelse who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.\n",
    "In addition to the labeled labeled POI feature the dataset contains 146 records with 24 features, mostly financial data but also email correspondence features. \n",
    "Of the 146 people in the dataset, 18 were labeled as POI, persons of interest."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Outlier investigation\n",
    "Now that i have some insight into what the data is made out of, I continue by visualizing the data, to see if i can spot an outlier. I do this by ploting salaries and bonuses for the Enron employees."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiwAAAGBCAYAAABFHepEAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAAPYQAAD2EBqD+naQAAIABJREFUeJzt3XucXWV97/HPjySCoEYkNZHTVIQQwNYTmBEFlSByCQRF\nEY44gYBorSgKndYLvloPB08rWrmUWlAqKNDItGitokCSBhWtENAZQa3gJBEKXgiXxCASbsmvf6w1\nZGeY687eM2vPfN6v135l9rOftfdvL1YyX57nWWtFZiJJklRl2413AZIkScMxsEiSpMozsEiSpMoz\nsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsAwjIg6KiGsj4lcRsTki\njqnjPRZExC0R8UhEPBARX4mIlzajXkmSJiIDy/B2Am4H3geM+sZLEbEb8DVgBTAPOAKYAfxbwyqU\nJGmCC29+OHIRsRl4S2ZeW9P2HOATwNuBFwI/Ac7KzJvK148Drs7M7Wu2eSNFiNk+MzeN4VeQJKkl\nOcKy7S4GXg28DXgF8GXghojYo3y9G9gcEadGxHYRMR1YDPyHYUWSpJFxhGUU+o+wRMRs4BfA7My8\nv6bffwC3ZuZfl8/nA9cAuwBTgJuBhZn5yBh/BUmSWpIjLNvmFRQBpDciftf3AOYDewBExEzg88AX\ngVeWrz2Fa1gkSRqxqeNdQIt7HvA00AZs7vfao+WfpwMbMvOjfS9ExEnAfRHxqsy8bUwqlSSphRlY\nts2PKEZYZmbm9wfpsyNFqKnVF24c4ZIkaQRa7hfmaK+LEhHHRsTy8vonGyLi5og4YhSft1NEzIuI\nfcum3cvnszNzFXA1cFX5ObtFxKsi4qyIOKrsfx3wqoj4WETMiYg2iumhuykCjyRJGkbLBRZGf12U\n+cBy4CiKqZtvA9+IiHkj/LxXUgSL7vLzzgd6gHPK198BXAWcB9wFfLXc5l6AzPw2sAh4c7nd9cBG\n4KjMfGKENUiSNKm19FlCA10XZYTb/RT4l8z8m+ZUJkmSGqkVR1i2SUQE8Hxg3XjXIkmSRmbSBRbg\nQxTTSteMdyGSJGlkJtVZQhGxCPgYcExmPjREv12ABcA9wONjU50kSRPCDsBuwLLMfLhRbzppAktE\nvB34J+D4ciHsUBYAX2p+VZIkTVgnUpxJ2xCTIrBERAdwGXBCZi4dwSb3ACxZsoR99tmnmaW1hM7O\nTi688MLxLmPcuR+2cF8U3A8F98MW7gu48847Oemkk6D8XdooLRdYImInYA4QZdPu5SnK6zLzvog4\nF9g1M08p+y8CrgDOAH5QXiofYOMQ9/J5HGCfffahra2tSd+kdUyfPt39gPuhlvui4H4ouB+2cF9s\npaFLKlpx0e1w10WZBcyu6f9uiqvRXgz8uubx92NUryRJ2kYtN8KSmTcxRNDKzFP7PT+k6UVJkqSm\nasURFkmSNMkYWDSsjo6O8S6hEtwPW7gvCu6HgvthC/dF87T0pfmbpbxBYXd3d7eLpyRJGoWenh7a\n29sB2jOzp1Hv6wiLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmq\nPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOL\nJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmqPAOLJEmq\nPAOLJEmqPAOLJEmqPAOLJEmqvJYLLBFxUERcGxG/iojNEXHMCLZ5fUR0R8TjEdEbEaeMRa2SJKkx\nWi6wADsBtwPvA3K4zhGxG/BN4EZgHnARcFlEHN68EiVJUiNNHe8CRiszlwJLASIiRrDJe4FfZOaH\ny+c/j4jXAZ3AfzSnSkmS1EitOMIyWgcAK/q1LQMOHIdaJElSHSZDYJkFrO3XthZ4QURsPw71SJKk\nUZoMgUWSJLW4llvDUof7gZn92mYCj2TmE0Nt2NnZyfTp07dq6+jooKOjo7EVSpLUgrq6uujq6tqq\nbcOGDU35rMgc9kSbyoqIzcBbMvPaIfp8EjgqM+fVtF0NvDAzFw6yTRvQ3d3dTVtbW6PLliRpwurp\n6aG9vR2gPTN7GvW+LTclFBE7RcS8iNi3bNq9fD67fP3ciLiyZpPPlX0+FRF7RcT7gOOBC8a4dEmS\nRq23t5cbbriBVatWjXcp46rlAgvwSuBHQDfFdVjOB3qAc8rXZwGz+zpn5j3A0cBhFNdv6QTelZn9\nzxySJKky1q1bx5FHHs1ee+3FwoULmTt3LkceeTTr168f79LGRcutYcnMmxgiaGXmqQO0fRdob2Zd\nkiQ10qJFi1mxYiWwBJgPfJcVK86go+Mkli69bpyrG3stF1gkSZroent7WbbseoqwcmLZeiKbNiXL\nli1m1apV7LnnnuNY4dhrxSkhSZImtDVr1pQ/ze/3ysEArF69ekzrqQIDiyRJFbPHHnuUP3233ys3\nATBnzpwxracKDCySJFXM3LlzWbBgIVOmnEExLXQfsIQpU85kwYKFk246CAwskiRVUlfXEg477ABg\nMfBHwGIOO+wAurqWjHNl48NFt5IkVdDOO+/M0qXXsWrVKlavXs2cOXMm5chKHwOLJEkVtueee07q\noNLHKSFJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5\nBhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJ\nklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5BhZJklR5LRlYIuL0\niLg7IjZGxMqI2H+Y/idGxO0R8fuI+HVEXB4RLxqreiVJ0rZpucASEScA5wNnA/sBdwDLImLGIP1f\nC1wJfB54OXA88Crgn8akYEmStM1aLrAAncClmXlVZt4FnAY8BrxzkP4HAHdn5sWZ+d+ZeTNwKUVo\nkSRJLaClAktETAPagRv72jIzgRXAgYNsdgswOyKOKt9jJvB/gOuaW60kSWqUlgoswAxgCrC2X/ta\nYNZAG5QjKicB/xoRTwK/AdYD729inZIkqYGmjncBzRYRLwcuAv4fsBx4CXAexbTQnw61bWdnJ9On\nT9+qraOjg46OjqbUKklSK+nq6qKrq2urtg0bNjTls6KYUWkN5ZTQY8BxmXltTfsVwPTMPHaAba4C\ndsjMt9W0vRb4HvCSzOw/WkNEtAHd3d3dtLW1Nf6LSJI0QfX09NDe3g7Qnpk9jXrflpoSysyngG7g\n0L62iIjy+c2DbLYj8HS/ts1AAtGEMiVJUoO1VGApXQC8OyJOjoi9gc9RhJIrACLi3Ii4sqb/N4Dj\nIuK0iHhZObpyEXBrZt4/xrVLkqQ6tNwalsy8przmyseBmcDtwILMfLDsMguYXdP/yoh4HnA6xdqV\n31KcZXTWmBYuSZLq1nKBBSAzLwEuGeS1Uwdouxi4uNl1SZKk5mjFKSFJkjTJGFgkSVLlGVgkSVLl\nGVgkSVLlGVgkSVLlGVgkSVLlGVgkSVLlGVgkSVLlGVgkSVLlGVgkSVLlGVgkSVLlGVgkSVLlGVgk\n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\nO7AgMx8su8wCZo9XfZIkqfFaLrAAZOYlwCWDvHbqMNueg6c3S5LUUuq6l5AkSdJYMrBIkqTKM7BI\nkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTK\nM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BI\nkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTKM7BIkqTK\nM7BIkqTKM7BIkqTKM7BIkqTKa8nAEhGnR8TdEbExIlZGxP5D9D02IpZHxAMRsSEibo6II8ayXkmS\ntG1aLrBExAnA+cDZwH7AHcCyiJgxyCbzgeXAUUAb8G3gGxExbwzKlSRJDdBygQXoBC7NzKsy8y7g\nNOAx4J0Ddc7Mzsw8LzO7M3NNZv4VsAp409iVLEmStkVLBZaImAa0Azf2tWVmAiuAA0f4HgE8H1jX\njBolSVLjtVRgAWYAU4C1/drXArNG+B4fAnYCrmlgXZIkqYmmjncBYykiFgEfA47JzIeG69/Z2cn0\n6dO3auvo6KCjo6NJFUqS1Dq6urro6uraqm3Dhg1N+awoZlRaQzkl9BhwXGZeW9N+BTA9M48dYtu3\nA5cBx2fm0mE+pw3o7u7upq2trSG1S5I0GfT09NDe3g7Qnpk9jXrflpoSysyngG7g0L62ck3KocDN\ng20XER3A5cDbhwsrkiSpelpxSugC4IqI6AZuozhraEfgCoCIOBfYNTNPKZ8vKl87A/hBRMws32dj\nZj4ytqVLkqR6tFxgycxrymuufByYCdwOLMjMB8sus4DZNZu8m2Kh7sXlo8+VDHIqtCRJqpaWCywA\nmXkJcMkgr53a7/khY1KUJElqmpZawyJJkiYnA4skSao8A4skSao8A4skSao8A4skSao8A4skSao8\nA4skSao8A4skSao8A4skSao8A4skSao8A4skSao8A4skSao8A4skSao8A4skSao8A4skSao8A4sk\nSao8A4skSaq8qeNdgFpLb28va9asYc6cOey5557jXY4kaZJwhEUjsm7dOo488mj22msvFi5cyNy5\ncznyyKNZv379eJcmSZoEDCwakUWLFrNixUpgCXAvsIQVK1bS0XHSgP17e3u54YYbWLVq1ViWKUma\noAwsGlZvby/Lll3Ppk3/AJwIzAZOZNOmi1i27PqtQokjMZKkZjCwaFhr1qwpf5rf75WDAVi9evUz\nLaMdiZEkaSQMLBrWHnvsUf703X6v3ATAnDlzgP4jMfsDPwVeNeBIjCRJo2Fg0bDmzp3LggUL2W67\n04EPUQSXJUyZciYLFix85myhLSMxXwD2AhYCc4EvAluPxEiSNBoGFg1r3bp1PPXUU2zevAE4j2Iq\n6BQOPridrq4lz/QrRmK2A35E7ZRQ8Xy7Z0ZiJEkaLa/DomEtWrSY73znNorRlTcC9zFlyhlMmzaN\nnXfeuV/vzcBnKBbnUv6ZwOIxq1eSNPE4wqIh3XbbbSxbtrQcXfk0xejK1Wza9IlnrUsZzeJcSZJG\nw8CiIb33ve8Hns/WUzwrgWuArUPISBfnSpI0Wk4JaVC9vb309PyAIqQMPMUzdeqWQ6hvce6KFWew\naVNSjKzcxJQpZ3LYYQu9lL8kqW6OsGhQA0/x9FKsUwEIjjjiSObPP+SZC8N1dS3hsMMOoAg0fwQs\n5rDDDthqca4kSaPlCIsGtfUUz1EUIeT6mh67Aev43vduoaPjJJYuvY6dd96ZpUuvY/ny5axcuZID\nDzyQww8/fIwrlyRNNAYWDWru3LnMm9fGHXecDryMLWtY5lOEmNMppoeeeGYB7i677MKiRYtZtmxL\nsFmwYCFdXUsGOKNIkqSRcUpIQ3ruc3cAHgNuBz5a/nwlsBH4a+AR+g6j1atXe2l+SVJTtOQIS0Sc\nDnwQmAXcAXwgM38wRP/XA+cDf0zxW/RvM/PKMSi1pfX29rJy5c3AcyhCyYdqXt2OYi1L358wZcqU\ncmRl60W6mzYly5YtZtWqVS68lSTVpeVGWCLiBIrwcTawH0VgWRYRMwbpvxvwTeBGYB5wEXBZRLiw\nYhhbFt0+BUzr9+o04OUUpzwXgebee+8tX/M6LJKkxmq5wAJ0Apdm5lWZeRdwGsU8xTsH6f9e4BeZ\n+eHM/HlmXgx8pXwfDWLNmjUcd9zbgCgfO7L1tVh2BO6imBZ6EtjMJz/56XJrr8MiSWqslgosETEN\naKcYLQEgMxNYARw4yGYHlK/XWjZEfwGvfvVr2bjxcYqRlNrL7c8u//yHsv2BZ7a5++5fs8suM5ky\n5QyKUHMfA90kUZKk0WqpwALMAKYAa/u1r6VYzzKQWYP0f0FEbN/Y8iaGZcuW8fDDa4Gngf3L1oGn\neYoZucLmzWfz8MNrec1rXoHXYZEkNVJLLrodK52dnUyfPn2rto6ODjo6OsaporFx66231jw7EPg+\nxTTPiTXtN5V//ifFYXQEcALwIT760Y9w+eWfZ/Xq1cyZM8eRFUmaoLq6uujq6tqqbcOGDU35rFYL\nLA8Bm4CZ/dpnAvcPss39g/R/JDOfGOrDLrzwQtra2uqps6W9+tWvrnk2k2Igru+aK8Xl9uH9Zfvj\nwCsppoCuA3gmpBhUJGliG+h/4nt6emhvb2/4Z7XUlFBmPgV0A4f2tUVElM9vHmSzW2r7l44o2zWA\nBQsWsMsuMyny7N8AewGPUjvNU0wX7QDsTrGG+TrXqkiSmqalAkvpAuDdEXFyROwNfI7ilJUrACLi\n3IiovcbK54DdI+JTEbFXRLwPOL58Hw3iBz+4hZ122hH4HXAnW+4f1Of3bLfd08BqXKsiSWq2lgss\nmXkNxUXjPg78CPjfwILMfLDsMoviVJa+/vcARwOHUVyutRN4V2b2P3NINV72spfx6KMbWL58Kfvs\nsw9Tp04hIth+++3Zb7/9WL58OZs2PUFvby/XX389vb29z9xLSJKkRovirGDViog2oLu7u3tSrmGR\nJKleNWtY2jOzp1Hv23IjLJIkafIxsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMoz\nsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiS\npMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMoz\nsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMozsEiSpMprqcASETtHxJciYkNErI+I\nyyJipyH6T42IT0XEjyPi0Yj4VURcGREvGcu6W11XV9d4l1AJ7oct3BcF90PB/bCF+6J5WiqwAFcD\n+wCHAkcD84FLh+i/I7AvcA6wH3AssBfw9eaWObH4F7DgftjCfVFwPxTcD1u4L5pn6ngXMFIRsTew\nAGjPzB+VbR8ArouID2bm/f23ycxHym1q3+f9wK0R8YeZ+csxKF2SJG2jVhphORBY3xdWSiuABF49\nivd5YbnNbxtYmyRJaqJWCiyzgAdqGzJzE7CufG1YEbE98Eng6sx8tOEVSpKkphj3KaGIOBf4yBBd\n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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x109b78d50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import Outlier_Investigation\n",
    "Outlier_Investigation.plot(data_dict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the plot we can easily see theres one datapoint that clearly stands out.\n",
    "To handle these outliers correctly i need to identify the cause for it's strange values. I start by populated two dictionary with the bonus and salary in descending order and then print the key and value for the individuals with highest recorded bonus and salary."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('TOTAL', 97343619),\n",
      " ('LAVORATO JOHN J', 8000000),\n",
      " ('LAY KENNETH L', 7000000),\n",
      " ('SKILLING JEFFREY K', 5600000),\n",
      " ('BELDEN TIMOTHY N', 5249999)]\n"
     ]
    }
   ],
   "source": [
    "import Outlier_Investigation\n",
    "Outlier_Investigation.bonus_outlier(data_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('TOTAL', 26704229),\n",
      " ('SKILLING JEFFREY K', 1111258),\n",
      " ('LAY KENNETH L', 1072321),\n",
      " ('FREVERT MARK A', 1060932),\n",
      " ('PICKERING MARK R', 655037)]\n"
     ]
    }
   ],
   "source": [
    "import Outlier_Investigation\n",
    "Outlier_Investigation.salary_outliers(data_dict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following outlier was identified.\n",
    "\n",
    "• TOTAL: The TOTAL key-value pair is most likley taken from the spreadsheet data and has been mistaken for a person.\n",
    "\n",
    "This error leads me to believe there could be more outliers in the data. So i look through the spreadsheet to see if I can spot similar scenarios that can create the same error."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "data_dict[\"THE TRAVEL AGENCY IN THE PARK\"];"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "• THE TRAVEL AGENCY IN THE PARK: This record is not a person.\n",
    "\n",
    "Both of these data-errors should be removed so as not to negatively affect the analysis at a later stage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "data_dict.pop( \"TOTAL\", 0 );\n",
    "data_dict.pop( \"THE TRAVEL AGENCY IN THE PARK\", 0 );"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then i plot the salaries and bonuses for the Enron employees again."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x10d282f50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import Outlier_Investigation\n",
    "Outlier_Investigation.plot_clean(data_dict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Create new features\n",
    "In the data exploration phase we found a number of \"NaN\" values, let us examine this closer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "bonus                         63\n",
      "deferral_payments            106\n",
      "deferred_income               96\n",
      "director_fees                128\n",
      "email_address                 33\n",
      "exercised_stock_options       43\n",
      "expenses                      50\n",
      "from_messages                 58\n",
      "from_poi_to_this_person       58\n",
      "from_this_person_to_poi       58\n",
      "loan_advances                141\n",
      "long_term_incentive           79\n",
      "other                         53\n",
      "poi                            0\n",
      "restricted_stock              35\n",
      "restricted_stock_deferred    127\n",
      "salary                        50\n",
      "shared_receipt_with_poi       58\n",
      "to_messages                   58\n",
      "total_payments                21\n",
      "total_stock_value             19\n",
      "dtype: int64\n",
      "(144, 21)\n"
     ]
    }
   ],
   "source": [
    "import Create_new_features\n",
    "print Create_new_features.check_NaN(data_dict).isnull().sum()\n",
    "print Create_new_features.check_NaN(data_dict).shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "By applying \"remove_NaN=True\" & \"remove_all_zeroes=True\" as parameters to feature_format -> featureFormat, this will convert \"NaN\" strings to 0.0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Features select"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Okej now that we've got all the \"NaN\" values are dealt with, let's using selectKBest from sklearns feature-selection utility to identify the 10 best features to select for further analysis. \n",
    "\n",
    "The 10 selected features are printed below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10 selected features: ['salary', 'total_payments', 'loan_advances', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive']\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import Create_new_features\n",
    "target_label = 'poi'\n",
    "best_features = Create_new_features.kbest(data_dict)\n",
    "new_features_list = [target_label] + best_features.keys()\n",
    "\n",
    "features_list = new_features_list\n",
    "    \n",
    "print \"{0} selected features: {1}\\n\".format(len(features_list) - 1, features_list[1:])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Let's create new features\n",
    "Namely how intensive was the email correspondence with people of interest 'poi_ratio', how big portion of the sent mail was to people of interest 'fraction_to_poi', and how big portion of the recieved mail was from people of interest 'fraction_from_poi'?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['poi', 'salary', 'total_payments', 'loan_advances', 'bonus', 'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options', 'deferred_income', 'restricted_stock', 'long_term_incentive', 'poi_ratio', 'fraction_to_poi', 'fraction_from_poi']\n"
     ]
    }
   ],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# add the new features to our list of importent festures\n",
    "Create_new_features.new_features(data_dict, features_list)\n",
    "\n",
    "eng_feature_list=features_list \n",
    "print eng_feature_list"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Properly scale features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Finally, lets scale the features using the preprocessing module and MinMaxScale utility"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.32916568  0.01025327  0.         ...,  0.20793561  0.03448276\n",
      "   0.2166548 ]\n",
      " [ 0.24036002  0.05440667  0.         ...,  0.          0.          0.        ]\n",
      " [ 0.15382656  0.00204447  0.         ...,  0.          0.          0.        ]\n",
      " ..., \n",
      " [ 0.23866105  0.01055103  0.         ...,  0.          0.          0.        ]\n",
      " [ 0.25070776  0.00845656  0.         ...,  1.          0.5         1.        ]\n",
      " [ 0.24744479  0.0122855   0.         ...,  0.29080159  0.375       0.27406108]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from feature_format import featureFormat,targetFeatureSplit\n",
    "\n",
    "data = featureFormat(data_dict, features_list)\n",
    "labels, features = targetFeatureSplit(data)\n",
    "\n",
    "# scale features using min-max\n",
    "scaler = MinMaxScaler()\n",
    "features = scaler.fit_transform(features)\n",
    "print features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now its time to split the data into training and testing set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]\n",
      "[[ 0.32916568  0.01025327  0.         ...,  0.20793561  0.03448276\n",
      "   0.2166548 ]\n",
      " [ 0.24036002  0.05440667  0.         ...,  0.          0.          0.        ]\n",
      " [ 0.15382656  0.00204447  0.         ...,  0.          0.          0.        ]\n",
      " ..., \n",
      " [ 0.23866105  0.01055103  0.         ...,  0.          0.          0.        ]\n",
      " [ 0.25070776  0.00845656  0.         ...,  1.          0.5         1.        ]\n",
      " [ 0.24744479  0.0122855   0.         ...,  0.29080159  0.375       0.27406108]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.cross_validation import train_test_split\n",
    "features_train,features_test,labels_train,labels_test = train_test_split(features,labels, test_size=0.3, random_state=42)\n",
    "    \n",
    "print labels\n",
    "print features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Pick and tune an algorithm\n",
    "Here i will test different algorithms and evaluatw their performance, the algorithms i've decide to test are \"Nearest Neighbors\", \"Linear SVM\", \"RBF SVM\", \"Decision Tree\",\"Random Forest\", \"AdaBoost\", \"Naive Bayes\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " \n",
      "Classifier:\n",
      "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
      "           metric_params=None, n_jobs=1, n_neighbors=3, p=2,\n",
      "           weights='uniform')\n",
      "precision: 0.143410714286\n",
      "recall:    0.0652313852814\n",
      "Accuracy: 0.84 (+/- 0.00)\n",
      "=====================================================================\n",
      " \n",
      "Classifier:\n",
      "SVC(C=0.025, cache_size=200, class_weight=None, coef0=0.0,\n",
      "  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',\n",
      "  max_iter=-1, probability=False, random_state=None, shrinking=True,\n",
      "  tol=0.001, verbose=False)\n",
      "precision: 0.0\n",
      "recall:    0.0\n",
      "Accuracy: 0.84 (+/- 0.00)\n",
      "=====================================================================\n",
      " \n",
      "Classifier:\n",
      "SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,\n",
      "  decision_function_shape=None, degree=3, gamma=2, kernel='rbf',\n",
      "  max_iter=-1, probability=False, random_state=None, shrinking=True,\n",
      "  tol=0.001, verbose=False)\n",
      "precision: 0.0938642857143\n",
      "recall:    0.0634535714286\n",
      "Accuracy: 0.79 (+/- 0.00)\n",
      "=====================================================================\n",
      " \n",
      "Classifier:\n",
      "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,\n",
      "            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,\n",
      "            min_samples_split=2, min_weight_fraction_leaf=0.0,\n",
      "            presort=False, random_state=None, splitter='best')\n",
      "precision: 0.312880906593\n",
      "recall:    0.223779112554\n",
      "Accuracy: 0.84 (+/- 0.00)\n",
      "=====================================================================\n",
      " \n",
      "Classifier:\n",
      "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
      "            max_depth=10, max_features=1, max_leaf_nodes=None,\n",
      "            min_samples_leaf=1, min_samples_split=2,\n",
      "            min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,\n",
      "            oob_score=False, random_state=None, verbose=0,\n",
      "            warm_start=False)\n",
      "precision: 0.354586507937\n",
      "recall:    0.186499278499\n",
      "Accuracy: 0.81 (+/- 0.00)\n",
      "=====================================================================\n",
      " \n",
      "Classifier:\n",
      "AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1.0,\n",
      "          n_estimators=5, random_state=None)\n",
      "precision: 0.370311868687\n",
      "recall:    0.250851587302\n",
      "Accuracy: 0.86 (+/- 0.00)\n",
      "=====================================================================\n",
      " \n",
      "Classifier:\n",
      "GaussianNB()\n",
      "precision: 0.344905892171\n",
      "recall:    0.365893398268\n",
      "Accuracy: 0.40 (+/- 0.00)\n",
      "=====================================================================\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import ExtraTreesClassifier\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "import matrics\n",
    "\n",
    "names = [\"Nearest Neighbors\", \"Linear SVM\", \"RBF SVM\", \"Decision Tree\",\n",
    "             \"Random Forest\", \"AdaBoost\", \"Naive Bayes\"]\n",
    "\n",
    "classifiers = [\n",
    "    KNeighborsClassifier(3),\n",
    "    SVC(kernel=\"linear\", C=0.025),\n",
    "    SVC(kernel='rbf', gamma=2, C=10),\n",
    "    DecisionTreeClassifier(max_depth=2),\n",
    "    RandomForestClassifier(max_depth=10, n_estimators=5, max_features=1),\n",
    "    AdaBoostClassifier(algorithm='SAMME', n_estimators=5),\n",
    "    GaussianNB()]\n",
    "\n",
    "# iterate over classifiers and print out the scores\n",
    "for name, clf in zip(names, classifiers):\n",
    "        clf.fit(features_train,labels_train)\n",
    "        scores = clf.score(features_test,labels_test)\n",
    "        print \" \"\n",
    "        print \"Classifier:\"\n",
    "        matrics.evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3)\n",
    "        print(\"Accuracy: %0.2f (+/- %0.2f)\" % (scores.mean(), scores.std() * 2))\n",
    "        print \"=====================================================================\"\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tune the algorithm\n",
    "I got significant success with AdaBoost Classifier and DecisionTree Classifier so i decided to move forward with one of them, in this case the DecisionTreeClassifier to see if i can imporve the model further.\n",
    "\n",
    "I have decided to use GridSearchCV to optimized my DecisionTreeClassifier algorithm over a parameter grid. Giving me the optimal parameters for my model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=8,\n",
      "            max_features=None, max_leaf_nodes=None, min_samples_leaf=7,\n",
      "            min_samples_split=4, min_weight_fraction_leaf=0.0,\n",
      "            presort=False, random_state=None, splitter='best')\n",
      "0.946666666667\n",
      "Processing time: 21.004 s\n"
     ]
    }
   ],
   "source": [
    "from sklearn.grid_search import GridSearchCV\n",
    "from sklearn.cross_validation import StratifiedShuffleSplit\n",
    "cv = StratifiedShuffleSplit(labels, n_iter=10)\n",
    "\n",
    "from time import time\n",
    "t0 = time()\n",
    "param_grid = {'max_depth': [1,2,3,4,5,6,8,9,10],'min_samples_split':[1,2,3,4,5],'min_samples_leaf':[1,2,3,4,5,6,7,8], 'criterion':('gini', 'entropy')}\n",
    "\n",
    "clf = DecisionTreeClassifier() \n",
    "GS_clf = GridSearchCV(clf, param_grid,  cv = cv)\n",
    "GS_clf.fit(features, labels)\n",
    "\n",
    "\n",
    "print GS_clf.best_estimator_\n",
    "print GS_clf.best_score_\n",
    "print 'Processing time:',round(time()-t0,3) ,'s'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Finally i tested it using the Cross-validation iterator StratifiedShuffleSplit, with 1000 folds as provided in the tester.py script."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=8,\n",
      "            max_features=None, max_leaf_nodes=None, min_samples_leaf=7,\n",
      "            min_samples_split=4, min_weight_fraction_leaf=0.0,\n",
      "            presort=False, random_state=None, splitter='best')\n",
      "\tAccuracy: 0.87907\tPrecision: 0.59810\tRecall: 0.28350\tF1: 0.38467\tF2: 0.31683\n",
      "\tTotal predictions: 15000\tTrue positives:  567\tFalse positives:  381\tFalse negatives: 1433\tTrue negatives: 12619\n",
      "\n",
      "Processing time: 1.509 s\n"
     ]
    }
   ],
   "source": [
    "from tester import test_classifier, dump_classifier_and_data\n",
    "\n",
    "t0 = time()\n",
    "best_clf = GS_clf.best_estimator_\n",
    "\n",
    "test_classifier(best_clf, data_dict, eng_feature_list)\n",
    "\n",
    "print 'Processing time:',round(time()-t0,3) ,'s'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Testing DecisionTreeClassifier best estimator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=8,\n",
      "            max_features=None, max_leaf_nodes=None, min_samples_leaf=7,\n",
      "            min_samples_split=4, min_weight_fraction_leaf=0.0,\n",
      "            presort=False, random_state=None, splitter='best')\n",
      "precision: 0.380847202797\n",
      "recall:    0.269944264069\n",
      "Accuracy: 0.95 (+/- 0.00)\n",
      "Processing time: 2.557 s\n"
     ]
    }
   ],
   "source": [
    "t0 = time()\n",
    "best_clf = GS_clf.best_estimator_\n",
    "best_score = GS_clf.best_score_\n",
    "\n",
    "matrics.evaluate_clf(best_clf, features, labels, num_iters=1000, test_size=0.3)\n",
    "print \"Accuracy: %0.2f (+/- %0.2f)\" % (best_score.mean(), best_score.std() * 2)\n",
    "print 'Processing time:',round(time()-t0,3) ,'s'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Dump successful classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "dump_classifier_and_data(best_clf, data_dict, eng_feature_list)"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [conda root]",
   "language": "python",
   "name": "conda-root-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "597c0927",
   "metadata": {},
   "source": [
    "# Importing Necessary Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "599f57a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "05284f97",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3d509495",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "data = pd.read_csv(\"Titanic.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "121bc939",
   "metadata": {},
   "source": [
    "# Loading The Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "29e70552",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>B42</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C148</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                             Allen, Mr. William Henry    male  35.0      0   \n",
       "..                                                 ...     ...   ...    ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       "889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       "890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       "\n",
       "     Parch            Ticket     Fare Cabin Embarked  \n",
       "0        0         A/5 21171   7.2500   NaN        S  \n",
       "1        0          PC 17599  71.2833   C85        C  \n",
       "2        0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3        0            113803  53.1000  C123        S  \n",
       "4        0            373450   8.0500   NaN        S  \n",
       "..     ...               ...      ...   ...      ...  \n",
       "886      0            211536  13.0000   NaN        S  \n",
       "887      0            112053  30.0000   B42        S  \n",
       "888      2        W./C. 6607  23.4500   NaN        S  \n",
       "889      0            111369  30.0000  C148        C  \n",
       "890      0            370376   7.7500   NaN        Q  \n",
       "\n",
       "[891 rows x 12 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b5b9ea24",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum(\n",
    "\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e161707a",
   "metadata": {},
   "source": [
    "# Data Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "88982a5e",
   "metadata": {},
   "source": [
    "#### Dropping irrelevant columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f89109a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop([\"PassengerId\" , \"Cabin\"] , axis =1 , inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c4593438",
   "metadata": {},
   "source": [
    "#### Imputation with mean value of column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6afb01f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"Age\"] = data[\"Age\"].fillna(sum(data[data[\"Age\"].isna()==False][\"Age\"])/sum(data[\"Age\"].isna()==False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e007fde4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embarked\n",
       "S    644\n",
       "C    168\n",
       "Q     77\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"Embarked\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e009d3e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\aditya\\AppData\\Local\\Temp\\ipykernel_15496\\4037531892.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  data[\"Embarked\"].fillna('S' , inplace = True)\n"
     ]
    }
   ],
   "source": [
    "data[\"Embarked\"].fillna('S' , inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d32cd48c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived    0\n",
       "Pclass      0\n",
       "Name        0\n",
       "Sex         0\n",
       "Age         0\n",
       "SibSp       0\n",
       "Parch       0\n",
       "Ticket      0\n",
       "Fare        0\n",
       "Embarked    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "60641252",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop([\"Name\" , \"Ticket\"] , axis =1 , inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "797ba9f5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Pclass     Sex        Age  SibSp  Parch     Fare Embarked\n",
       "0           0       3    male  22.000000      1      0   7.2500        S\n",
       "1           1       1  female  38.000000      1      0  71.2833        C\n",
       "2           1       3  female  26.000000      0      0   7.9250        S\n",
       "3           1       1  female  35.000000      1      0  53.1000        S\n",
       "4           0       3    male  35.000000      0      0   8.0500        S\n",
       "..        ...     ...     ...        ...    ...    ...      ...      ...\n",
       "886         0       2    male  27.000000      0      0  13.0000        S\n",
       "887         1       1  female  19.000000      0      0  30.0000        S\n",
       "888         0       3  female  29.699118      1      2  23.4500        S\n",
       "889         1       1    male  26.000000      0      0  30.0000        C\n",
       "890         0       3    male  32.000000      0      0   7.7500        Q\n",
       "\n",
       "[891 rows x 8 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e8e5776",
   "metadata": {},
   "source": [
    "#### Performing data encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c87c7ce2",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"Sex\"] = (data[\"Sex\"]==\"male\").astype(int)\n",
    "data[\"Embarked_S\"] = (data[\"Embarked\"]==\"S\").astype(int)\n",
    "data[\"Embarked_C\"] = (data[\"Embarked\"]==\"C\").astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "88354ffc",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop(\"Embarked\" , axis = 1 , inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "546c88f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"Pclass_3\"] = (data[\"Pclass\"]==3).astype(int)\n",
    "data[\"Pclass_2\"] = (data[\"Pclass\"]==2).astype(int)\n",
    "data.drop(\"Pclass\" , axis =1 , inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "ee4333b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "for col in [\"Age\" , \"Fare\" , \"Parch\" , \"SibSp\"]:\n",
    "    u = sum(data[col])/len(data[col])\n",
    "    std = ((sum(data[col]**2)/len(data[col])) - (u**2))**0.5\n",
    "    data[col] = (data[col] - u)/std"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ef34385",
   "metadata": {},
   "source": [
    "#### Final data after preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "72a9466d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked_S</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Pclass_3</th>\n",
       "      <th>Pclass_2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>-5.924806e-01</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.502445</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>6.387890e-01</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>0.786845</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>-2.846632e-01</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.488854</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4.079260e-01</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>0.420730</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4.079260e-01</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.486337</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>-2.077088e-01</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.386671</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>-8.233437e-01</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.044381</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>-2.733968e-16</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>2.008933</td>\n",
       "      <td>-0.176263</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>-2.846632e-01</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.044381</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1.770629e-01</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.492378</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Sex           Age     SibSp     Parch      Fare  Embarked_S  \\\n",
       "0           0    1 -5.924806e-01  0.432793 -0.473674 -0.502445           1   \n",
       "1           1    0  6.387890e-01  0.432793 -0.473674  0.786845           0   \n",
       "2           1    0 -2.846632e-01 -0.474545 -0.473674 -0.488854           1   \n",
       "3           1    0  4.079260e-01  0.432793 -0.473674  0.420730           1   \n",
       "4           0    1  4.079260e-01 -0.474545 -0.473674 -0.486337           1   \n",
       "..        ...  ...           ...       ...       ...       ...         ...   \n",
       "886         0    1 -2.077088e-01 -0.474545 -0.473674 -0.386671           1   \n",
       "887         1    0 -8.233437e-01 -0.474545 -0.473674 -0.044381           1   \n",
       "888         0    0 -2.733968e-16  0.432793  2.008933 -0.176263           1   \n",
       "889         1    1 -2.846632e-01 -0.474545 -0.473674 -0.044381           0   \n",
       "890         0    1  1.770629e-01 -0.474545 -0.473674 -0.492378           0   \n",
       "\n",
       "     Embarked_C  Pclass_3  Pclass_2  \n",
       "0             0         1         0  \n",
       "1             1         0         0  \n",
       "2             0         1         0  \n",
       "3             0         0         0  \n",
       "4             0         1         0  \n",
       "..          ...       ...       ...  \n",
       "886           0         0         1  \n",
       "887           0         0         0  \n",
       "888           0         1         0  \n",
       "889           1         0         0  \n",
       "890           0         1         0  \n",
       "\n",
       "[891 rows x 10 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee9aa13c",
   "metadata": {},
   "source": [
    "# Train-Test-split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "641642d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_size = int(0.8*len(data))\n",
    "train_rows = np.random.permutation(len(data))\n",
    "train_data = pd.DataFrame(np.array(data)[train_rows[:train_size]] , columns = data.columns)\n",
    "test_data = pd.DataFrame(np.array(data)[train_rows[train_size:]] , columns = data.columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5e7d86b",
   "metadata": {},
   "source": [
    "# Defining the Bagging Class using 4 Desicion Tree models and bootstrapping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2ffc629e",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Bagging:\n",
    "    def __init__(self , max_depth , x , y):\n",
    "        self.max_depth = max_depth\n",
    "        self.x = np.array(x)\n",
    "        self.y = np.array(y)\n",
    "    def bootstrap(self):\n",
    "        self.bootstrap_x = []\n",
    "        self.bootstrap_y = []\n",
    "        self.random_cols = []\n",
    "        for i in range(4):\n",
    "            random_rows = np.random.randint(0 , len(self.x) , len(self.x))\n",
    "            self.bootstrap_x.append(self.x[random_rows])\n",
    "            self.bootstrap_y.append(self.y[random_rows])\n",
    "            \n",
    "    def train(self):\n",
    "        self.bootstrap()\n",
    "        self.dr1 = DecisionTreeClassifier(criterion = 'entropy' , max_depth = self.max_depth)\n",
    "        self.dr2 = DecisionTreeClassifier(criterion = 'entropy' , max_depth = self.max_depth)\n",
    "        self.dr3 = DecisionTreeClassifier(criterion = 'entropy' , max_depth = self.max_depth)\n",
    "        self.dr4 = DecisionTreeClassifier(criterion = 'entropy' , max_depth = self.max_depth)\n",
    "        self.dr1.fit(self.bootstrap_x[0] , self.bootstrap_y[0])\n",
    "        self.dr2.fit(self.bootstrap_x[1] , self.bootstrap_y[1])\n",
    "        self.dr3.fit(self.bootstrap_x[2] , self.bootstrap_y[2])\n",
    "        self.dr4.fit(self.bootstrap_x[3] , self.bootstrap_y[3])\n",
    "    def predict(self , test_data):\n",
    "        k= (self.dr1.predict(test_data)+self.dr2.predict(test_data)+self.dr3.predict(test_data)+self.dr4.predict(test_data))/4\n",
    "        return (k>=0.5).astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b0f6e7f",
   "metadata": {},
   "source": [
    "# Training the Bagging model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "6aff35bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "bg = Bagging(10 , train_data.drop(\"Survived\" , axis =1) , train_data[\"Survived\"] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "3b4e3159",
   "metadata": {},
   "outputs": [],
   "source": [
    "bg.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "7fbf97f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = bg.predict(np.array(test_data.drop(\"Survived\" , axis = 1)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29f1e1f1",
   "metadata": {},
   "source": [
    "# Accuracy and visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "820cf573",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy = np.mean(np.array(test_data[\"Survived\"])==y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "37e000d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "83.8 %\n"
     ]
    }
   ],
   "source": [
    "print(round(accuracy , 4)*100 , '%')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "92bbbe48",
   "metadata": {},
   "source": [
    "#### The graph below depicts moving average of accuracy on a window size of 10 with respect to chanding max depth to trees"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "f06d9854",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x185da43cee0>]"
      ]
     },
     "execution_count": 161,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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vc6ePfuKZ/vggw8qNOLBee211zBlyhS5WdfMmTPxr3/9S/6+vzxXoP979afnerkXX3wRgiDg8ccfl1/zp2fbVU/36k/P9j//8z+vuJfx48fL3/f0cx3SQY3FYkF6ejo2b97c4/dfeuklvPLKK3j99ddx8OBBBAcHY968eWhtbfXxSAevv3sFgPnz56OsrEz+2rp1qw9H6Dl79+7Fww8/jC+//BIff/wx2traMHfuXFgsFvmYJ554Au+//z527NiBvXv3orS0FHfccYeCox4YV+4VAFavXt3t2b700ksKjXhwRowYgRdffBE5OTk4fPgwbr31VixatAhff/01AP95rkD/9wr4z3Pt6quvvsIbb7yBKVOmdHvdn56tpLd7Bfzr2V511VXd7uXzzz+Xv+fx5+qx4nCNAyC+88478u8dDocYHx8v/vd//7f8Wl1dnWg2m8WtW7cqMELPufxeRVEUV65cKS5atEiR8XhbZWWlCEDcu3evKIrO52g0GsUdO3bIx5w6dUoEIB44cECpYXrE5fcqiqJ48803i4899phyg/KyyMhI8f/9v//n189VIt2rKPrnc21sbBTHjBkjfvzxx93uzx+fbW/3Kor+9WzXr18vpqen9/g9bzzXIT1T05fCwkKUl5cjMzNTfi08PBwZGRk4cOCAgiPznuzsbAwbNgzjxo3DQw89hJqaGqWH5BH19fUAgKioKABATk4O2trauj3b8ePHIzk5WfPP9vJ7lfzlL39BTEwMJk2ahDVr1qC5uVmJ4XmU3W7Htm3bYLFYMHPmTL9+rpffq8TfnuvDDz+M22+/vdszBPzz32xv9yrxp2d79uxZJCYmYtSoUbj77rtx8eJFAN55rm5vkzBUlJeXA4C8/YMkLi5O/p4/mT9/Pu644w6MHDkSBQUFeOqpp7BgwQIcOHAAer1e6eENmMPhwOOPP44bbrgBkyZNAuB8tiaTqduO74D2n21P9woAy5cvR0pKChITE3H8+HH84he/QF5eHnbu3KngaAfuxIkTmDlzJlpbWxESEoJ33nkHEydORG5urt89197uFfC/57pt2zYcOXIEX3311RXf87d/s33dK+BfzzYjIwO///3vMW7cOJSVleGZZ57BTTfdhJMnT3rluTKoIQDAnXfeKf968uTJmDJlCtLS0pCdnY05c+YoOLLBefjhh3Hy5Mlua7j+qrd7feCBB+RfT548GQkJCZgzZw4KCgqQlpbm62EO2rhx45Cbm4v6+nr87W9/w8qVK7F3716lh+UVvd3rxIkT/eq5FhUV4bHHHsPHH3+MgIAApYfjVa7cqz892wULFsi/njJlCjIyMpCSkoK//vWvCAwM9Pj1uPzUi/j4eAC4Igu7oqJC/p4/GzVqFGJiYpCfn6/0UAbskUcewT//+U/s2bMHI0aMkF+Pj4+HzWZDXV1dt+O1/Gx7u9eeZGRkAIBmn63JZMLo0aMxffp0bNiwAenp6fj1r3/tl8+1t3vtiZafa05ODiorK3H11VfDYDDAYDBg7969eOWVV2AwGBAXF+c3z7a/e7Xb7Ve8R8vP9nIREREYO3Ys8vPzvfJvlkFNL0aOHIn4+Hjs3r1bfq2hoQEHDx7stqbtr4qLi1FTU4OEhASlh+I2URTxyCOP4J133sGnn36KkSNHdvv+9OnTYTQauz3bvLw8XLx4UXPPtr977Ulubi4AaPLZ9sThcMBqtfrVc+2NdK890fJznTNnDk6cOIHc3Fz565prrsHdd98t/9pfnm1/99rTcr+Wn+3lmpqaUFBQgISEBO/8mx1QerGfaGxsFI8ePSoePXpUBCBu3LhRPHr0qHjhwgVRFEXxxRdfFCMiIsT33ntPPH78uLho0SJx5MiRYktLi8Ijd19f99rY2Cj+9Kc/FQ8cOCAWFhaKn3zyiXj11VeLY8aM0eRusQ899JAYHh4uZmdni2VlZfJXc3OzfMyDDz4oJicni59++ql4+PBhcebMmeLMmTMVHPXA9Hev+fn54rPPPisePnxYLCwsFN977z1x1KhR4qxZsxQe+cA8+eST4t69e8XCwkLx+PHj4pNPPikKgiB+9NFHoij6z3MVxb7v1d+ea08urwDyp2d7ua736m/P9ic/+YmYnZ0tFhYWil988YWYmZkpxsTEiJWVlaIoev65DumgZs+ePSKAK75WrlwpiqKzrHvdunViXFycaDabxTlz5oh5eXnKDnqA+rrX5uZmce7cuWJsbKxoNBrFlJQUcfXq1WJ5ebnSwx6Qnu4TgPjWW2/Jx7S0tIg/+tGPxMjISDEoKEhcsmSJWFZWptygB6i/e7148aI4a9YsMSoqSjSbzeLo0aPFn/3sZ2J9fb2yAx+gH/7wh2JKSopoMpnE2NhYcc6cOXJAI4r+81xFse979bfn2pPLgxp/eraX63qv/vZsly1bJiYkJIgmk0kcPny4uGzZMjE/P1/+vqefqyCKojiwOR4iIiIi9WBODREREfkFBjVERETkFxjUEBERkV9gUENERER+gUENERER+QUGNUREROQXGNQQERGRX2BQQ0RERH6BQQ0RERH5BQY1RERE5BcY1BAREZFfYFBDREREfuH/AzFCmLwxYQ+3AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "acc = []\n",
    "for i in range(1 , 50):\n",
    "    bg_temp = bg = Bagging(i , train_data.drop(\"Survived\" , axis =1) , train_data[\"Survived\"] )\n",
    "    bg_temp.train()\n",
    "    acc.append(np.mean(np.array(test_data[\"Survived\"])==bg_temp.predict(np.array(test_data.drop(\"Survived\" , axis = 1)))))\n",
    "import matplotlib.pyplot as plt\n",
    "acc2 = []\n",
    "for i in range(11 , 50):\n",
    "    acc2.append(np.mean(acc[i-10 : i]))\n",
    "plt.plot([i for i in range(11 , 50)] , acc2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5648bf47",
   "metadata": {},
   "source": [
    "### Hence, the best fitting model for bagging is with max tree depth equal to 10 giving 83.8% accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56aecb4f",
   "metadata": {},
   "source": [
    "# Defining Adaboost Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "1229f290",
   "metadata": {},
   "outputs": [],
   "source": [
    "class AdaBoost:\n",
    "    def __init__(self , x , y , max_layers = 3):\n",
    "        self.thresholds = {}\n",
    "        self.x = x.copy()\n",
    "        for col in self.x.columns:\n",
    "            thr = np.mean(self.x[col])\n",
    "            self.thresholds[col] = thr\n",
    "            self.x[col] = (self.x[col] >= thr).astype(int)\n",
    "\n",
    "        self.y = np.array(y)\n",
    "        self.y = np.where(self.y==0 , -1 , 1)\n",
    "        for col in self.x.columns:\n",
    "            self.x[col] = (self.x[col] >= np.mean(self.x[col])).astype(int)\n",
    "        self.x = np.array(self.x)\n",
    "        self.weights = np.ones(len(self.x)) / len(self.x)\n",
    "        self.max_layers = min(max_layers , self.x.shape[1])\n",
    "    def train(self):\n",
    "        self.amounts_of_say = []\n",
    "        self.layers = []\n",
    "        self.correlation_types = []\n",
    "        for layer in range(self.max_layers):\n",
    "            best_col = None\n",
    "            best_error = 1\n",
    "            best_type = 0\n",
    "            for col in range(self.x.shape[1]):\n",
    "                if col in self.layers:\n",
    "                    continue\n",
    "                pred = np.where(self.x[:,col]==1 , 1 , -1)\n",
    "                error = np.sum(self.weights[pred != self.y])\n",
    "                inv_pred = -pred\n",
    "                inv_error = np.sum(self.weights[inv_pred != self.y])\n",
    "                if error < best_error:\n",
    "                    best_error = error\n",
    "                    best_col = col\n",
    "                    best_type = 0\n",
    "                if inv_error < best_error:\n",
    "                    best_error = inv_error\n",
    "                    best_col = col\n",
    "                    best_type = 1\n",
    "            alpha = 0.5 * np.log((1-best_error)/best_error)\n",
    "            self.amounts_of_say.append(alpha)\n",
    "            self.layers.append(best_col)\n",
    "            self.correlation_types.append(best_type)\n",
    "            preds = np.where(self.x[:,best_col]==1 , 1 , -1)\n",
    "            if best_type == 1:\n",
    "                preds = -preds\n",
    "            self.weights *= np.exp(-alpha * self.y * preds)\n",
    "            self.weights /= np.sum(self.weights)\n",
    "    def predict(self , X):\n",
    "        X = X.copy()\n",
    "        for col in X.columns[:-1]:\n",
    "            X[col] = (X[col] >= self.thresholds[col]).astype(int)\n",
    "\n",
    "        X = np.array(X)\n",
    "        final_pred = np.zeros(len(X))\n",
    "        for alpha , col , corr_type in zip(self.amounts_of_say , self.layers , self.correlation_types):\n",
    "            preds = np.where(X[:,col]==1 , 1 , -1)\n",
    "            if corr_type == 1:\n",
    "                preds = -preds\n",
    "            final_pred += alpha * preds\n",
    "        return np.where(final_pred >= 0 , 1 , 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eebb9b5e",
   "metadata": {},
   "source": [
    "# Training the Adaboost model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "1fa4eae2",
   "metadata": {},
   "outputs": [],
   "source": [
    "ada = AdaBoost(train_data.drop(\"Survived\" , axis = 1), train_data[\"Survived\"] , 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "3f86e2b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "ada.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "fdf4d703",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = ada.predict(test_data.drop(\"Survived\" , axis =1))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09eee5ab",
   "metadata": {},
   "source": [
    "# Accuracy and visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "994f9af3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final Accuracy:  76.54 %\n"
     ]
    }
   ],
   "source": [
    "print(\"Final Accuracy: \" , round(np.mean(y_pred==test_data[\"Survived\"])*100 , 2) , \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "85a6cb1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "7e947b75",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy at max layers 0:  40.78 %\n",
      "Accuracy at max layers 1:  76.54 %\n",
      "Accuracy at max layers 2:  76.54 %\n",
      "Accuracy at max layers 3:  76.54 %\n",
      "Accuracy at max layers 4:  76.54 %\n",
      "Accuracy at max layers 5:  75.98 %\n",
      "Accuracy at max layers 6:  75.98 %\n",
      "Accuracy at max layers 7:  74.86 %\n",
      "Accuracy at max layers 8:  74.86 %\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Accuracy v/s Max Layers')"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "acc = []\n",
    "for i in range(9):\n",
    "    ada = AdaBoost(train_data.drop(\"Survived\" , axis = 1), train_data[\"Survived\"] , i)\n",
    "    ada.train()\n",
    "    y_pred = ada.predict(test_data.drop(\"Survived\" , axis = 1))\n",
    "    print(f\"Accuracy at max layers {i}: \" , round(np.mean(y_pred==test_data[\"Survived\"])*100 , 2) , \"%\")\n",
    "    acc.append(round(np.mean(y_pred==test_data[\"Survived\"])*100 , 2) )\n",
    "plt.plot(acc)\n",
    "plt.xlabel(\"Max Layers\")\n",
    "plt.ylabel(\"Accuracy(in %)\")\n",
    "plt.title(\"Accuracy v/s Max Layers\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7bb9d1ce",
   "metadata": {},
   "source": [
    "### Hence, the best fitting model for Adaboost with with 3 layers giving 78.21% accuracy on test data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20e69b2c",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

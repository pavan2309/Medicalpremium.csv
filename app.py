import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt   

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
  
# loading in the model to predict on the data
pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@st.cache 
def load_data(nrows):
    df = pd.read_csv('Medicalpremium.csv')
    return df
  
def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(Age,Diabetes,BP,Transplants,ChronicDiseases,Height,Weight,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries):  
   
    prediction = classifier.predict(
        [[Age,Diabetes,BP,Transplants,ChronicDiseases,Height,Weight,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries]])
    print(prediction)
    return prediction
      
  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    
    
    with st.sidebar:
        selected = option_menu(
            menu_title = "menu",
            options=["Home","Predict","Contact", "Visualizations"],
        )

    if selected == "Home":
        st.title("PYAR Insurance Company")
        st.header("Mission :")
        st.markdown("**To be the most preferred choice of customers for General Insurance by building Relationships and grow profitably.**")
        st.header("Vision :")
        st.markdown("Leveraging technology to integrate people and processes.")
        st.markdown("To excel in service and performance.")
        st.markdown("To uphold the highest ethical standards in conducting our business.")
        st.header("What is Insurance?")
        st.markdown("Most people have some kind of insurance: for their car, their house, or even their life. Yet most of us don’t stop to think too much about what insurance is or how it works.Put simply, insurance is a contract, represented by a policy, in which a policyholder receives financial protection or reimbursement against losses from an insurance company. The company pools clients’ risks to make payments more affordable for the insured.Insurance policies are used to hedge against the risk of financial losses, both big and small, that may result from damage to the insured or their property, or from liability for damage or injury caused to a third party.")
        st.header("KeyTakeaways")
        st.markdown("Insurance is a contract (policy) in which an insurer indemnifies another against losses from specific contingencies or perils.")
        st.markdown("There are many types of insurance policies. Life, health, homeowners, and auto are the most common forms of insurance")
        st.markdown("The core components that make up most insurance policies are the deductible, policy limit, and premium.")
        st.header("How Insurance Works")
        st.markdown("A multitude of different types of insurance policies is available, and virtually any individual or business can find an insurance company willing to insure them—for a price. The most common types of personal insurance policies are auto, health, homeowners, and life. Most individuals in the United States have at least one of these types of insurance, and car insurance is required by law.Businesses require special types of insurance policies that insure against specific types of risks faced by a particular business. For example, a fast-food restaurant needs a policy that covers damage or injury that occurs as a result of cooking with a deep fryer. An auto dealer is not subject to this type of risk but does require coverage for damage or injury that could occur during test drives.")
        st.subheader("Important Note:")
        st.markdown("To select the best policy for you or your family, it is important to pay attention to the three critical components of most insurance policies: 1.deductible, 2.premium, and 3.policy limit.")
        

# =============================================================================
#         def load_lottiefile(filepath: str):
#             with open (filepath, "r") as f: 
#                 return json.load(f)
#         hello_lt = load_lottiefile(r'C:\Users\ANIRUDDHA\OneDrive\Desktop\anju\insurance\hello.json')
#         st_lottie(
#             hello_lt,
#             height=100,
#             width=100
#         )
# =============================================================================
    
    if selected == "Predict":
        
        html_temp = """
        <div style ="background-color:gray;padding:10px">
        <h1 style ="color:black;text-align:center;">Insurance Premium Prediction </h1>
        </div>
        """ 
        
        st.markdown(html_temp, unsafe_allow_html = True)
        col1, col2, col3 = st.columns(3)
        with col1:
            Age = st.number_input("Enter your age",min_value=1,max_value=100)
        with col2:
            st.header("")
        with col3:
            st.header("")
        
            
        col1, col2 = st.columns(2)
        with col1:
            Height = st.slider('Height(cm)', 10, 200, 1)
        with col2: 
            Weight = st.slider('Weight(kg)', 10, 150, 1) 
        st.header("Give your current health status")
        a = st.checkbox('Diabetes')
        if a:
            Diabetes = 1.0
        else:
            Diabetes = 0.0
        a = st.checkbox('Blood pressure')
        if a:
            BP = 1.0
        else:
            BP = 0.0
        a = st.checkbox('Transplants')
        if a:
            Transplants = 1.0
        else:
            Transplants = 0.0
        a = st.checkbox('Chronic Diseases')
        if a:
            ChronicDiseases = 1.0
        else:
            ChronicDiseases = 0.0
        a = st.checkbox('Known Allergies')
        if a:
            KnownAllergies = 1.0
        else:
            KnownAllergies = 0.0
        a = st.checkbox('History Of Cancer In Family')
        if a:
            HistoryOfCancerInFamily = 1.0
        else:
            HistoryOfCancerInFamily = 0.0
        col1, col2 = st.columns(2)
        with col1:
            NumberOfMajorSurgeries = st.number_input("Number Of Major Surgeries",min_value=0,max_value=10)
        with col2:
            st.header("")
    
        result =""
      
        if st.button("Predict"):
            result = prediction(float(Age), float(Diabetes), float(BP), float(Transplants), float(ChronicDiseases), float(Height), float(Weight), float(KnownAllergies), float(HistoryOfCancerInFamily), float(NumberOfMajorSurgeries))
            st.success('Hey! your premium health insurance price is Rs. {}'.format(result))
    if selected == "Contact": 
        st.title("PYAR Insurance Company")
        st.subheader("Reach us at:")
        col1, col2 = st.columns(2)
        with col1:
            st.text("abc: 1111111111")
        with col2:
            st.text("def: 2222222222")
        col1, col2 = st.columns(2)
        with col1:
            st.text("pqr: 3333333333")
        with col2:
            st.text("lmn: 4444444444")
        col1, col2 = st.columns(2)
        with col1:
            st.text("pyarinsurance@pyar.com")
        with col2:
            st.text("insurance@pyar.com")

    if selected == "Visualizations":
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        st.title("Visualizations of the Data")
        st.subheader("DataFrame")
        df = load_data(1000)
        st.write(df)
        from sklearn.ensemble import ExtraTreesClassifier
        X=df.drop('PremiumPrice',axis=1)
        y=df['PremiumPrice']
        model = ExtraTreesClassifier()
        model.fit(X,y)
        feat_imp =pd.Series(model.feature_importances_,index=X.columns)
        feat_imp.nlargest(10).plot(kind='barh')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Bar Plot")
        st.pyplot()

        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3,random_state=43)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        d_tree = DecisionTreeClassifier()
        d_tree.fit(X_train,Y_train)
        y_pred_tree= d_tree.predict(X_test)
        y_prob_tree =d_tree.predict_proba(X_test)

        rf = RandomForestClassifier()
        rf.fit(X_train,Y_train)
        y_pred_rf= rf.predict(X_test)
        y_prob_rf =rf.predict_proba(X_test)

        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm_tree = confusion_matrix(Y_test,y_pred_tree)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm_tree,annot=True)
        plt.xlabel("predicted")
        plt.ylabel("truth")
        st.subheader("Heatmap")
        st.pyplot()

        cm_rf = confusion_matrix(Y_test,y_pred_rf)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm_rf,annot=True)
        plt.xlabel("predicted")
        plt.ylabel("truth")
        st.subheader("Heatmap")
        st.pyplot()

        
        
     
if __name__=='__main__':
    main()

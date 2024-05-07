import pickle
import base64
from io import BytesIO
import numpy as np
import streamlit as str
import pandas as pd
import cv2
from streamlit_option_menu import option_menu
from PIL import Image
from streamlit_lottie import st_lottie
import json

#loading the models
heart_disease_model = pickle.load(open('D:/Minor Project/heart disease/heart_disease_model.sav', 'rb'))
parkinson_disease_model = pickle.load(open('D:/Minor Project/parkinson disease/parkinson_disease_model.sav','rb'))
diabetes_model = pickle.load(open('D:/Minor Project/diabetes/diabetes_model.sav','rb'))
cancer_model = pickle.load(open('D:/Minor Project/cancer/cancer_model.sav','rb'))
#brain_tumor_model = pickle.load(open('D:/Minor Project/brain tumor/brain_tumor_model.sav','rb'))



# sidebar for navigation
with str.sidebar:
    select = option_menu( 'Multiple Disease Prediction System',
                          ['Home Page','Heart Disease Prediction','Diabetes','Parkinson','Cancer','Brain Tumor'],
                         icons=['house','heart','activity','person-standing','asterisk','cloud-fill'],
                          default_index=0)

def diabetes_treatment_plan(data):
    # Add your personalized treatment plan generation logic here
        return "Your personalized treatment plan:\n1. Maintain a balanced diet\n2. Engage in regular physical activity\n3. Monitor blood sugar levels regularly\n4. Take prescribed medications as directed\n5. Follow up with your healthcare provider regularly"

def heart_disease_treatment_plan(data):
    # Add your personalized treatment plan generation logic here
    return "Your personalized treatment plan for heart disease:\n1. Adopt a heart-healthy diet\n2. Maintain a healthy weight\n3. Exercise regularly\n4. Quit smoking\n5. Limit alcohol intake"

def parkinson_disease_treatment_plan(data):
    return "Treatment Plan \n1. Medications: The doctor may prescribe medications to help manage the symptoms of Parkinson's Disease.\n2. Physical Therapy: Physical therapy can help improve mobility, flexibility, and balance.\n3. Lifestyle Changes: Making certain lifestyle changes, such as getting regular exercise and eating a healthy diet, can help manage symptoms.\n4. Surgery: In some cases, surgery may be recommended to help manage symptoms."

def  cancer_treatment_plan(data):
    return "Treatment Plan for Cancer:\nThe treatment plan for cancer may vary based on the stage and other factors. However, common treatments include surgery, chemotherapy, radiation therapy, hormone therapy, and targeted therapy. Your healthcare provider will recommend the best treatment plan for you based on your specific situation."

def get_download_link(treatment_plan):
    # Generate a download link for the treatment plan
    b64 = base64.b64encode(treatment_plan.encode()).decode()  # Encode treatment plan as base64
    href = f'<a href="data:file/txt;base64,{b64}" download="treatment_plan.txt">Download Treatment Plan</a>'
    return href
    


# Heart Disease Prediction
if(select =='Home Page'):
    #str.title("Multiple Disease Prediction Web App")
    # Load the Lottie animation JSON file
    with open('D:/Minor Project/icons/Animation - 1712175992472.json', 'r') as f:
        animation_json = json.load(f)
        str.markdown("<h1 style='text-align: center;'>Welcome to the Disease Prediction System</h1>", unsafe_allow_html=True)
    # Display the Lottie animation
    st_lottie(animation_json, height=200, loop=True)
    str.write("Welcome to our Multiple Disease Prediction Web App!")
    str.write("Our web application uses advanced machine learning models to predict various diseases including heart disease, Parkinson's disease, diabetes, cancer, and brain tumor.")
    str.write("By leveraging cutting-edge technologies such as convolutional neural networks (CNNs), support vector machines (SVM), logistic regression, k-nearest neighbors (KNN), random forest, and naive Bayes, we provide accurate predictions based on input data provided by users.")
    str.write("Whether you're looking to assess your health status or seeking early detection of potential diseases, our app offers a user-friendly interface to upload your data and receive instant predictions.")
    str.write("Our goal is to empower individuals to take proactive steps towards their health and well-being.Explore our app and discover how technology can support your journey to better health.")
    str.write("Please select a disease prediction from the sidebar.")


if(select == 'Heart Disease Prediction'):
    str.title("Heart Disease Prediction")

    #with open('D:/Minor Project/icons/Heart.json', 'r') as f:
        #animation_json = json.load(f)
    #st_lottie(animation_json, height=100, loop=True,)
    # User Input Fields
    col1,col2 = str.columns(2)
    with col1:
        age = str.text_input("Age")
        cp_mapping = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
        cp_input = str.selectbox("Chest Pain",["Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"])
        cp = cp_mapping[cp_input]
        chol = str.text_input("Serum Cholesterol Level",placeholder="mg/dl")
        rest_map = {"Normal":0,"Having ST-T wave abnormality":1  ,"Showing probable or definite left ventricular hypertrophy":2}
        rest_ip=str.selectbox("Resting electrocardiographic",["Normal","Having ST-T wave abnormality","Showing probable or definite left ventricular hypertrophy"])
        restecg = rest_map[rest_ip]
        exang_map = {"Yes":1,"No":0}
        exang_ip = str.selectbox("Exercise-induced angina",["Yes","No"])
        exang = exang_map[exang_ip] 
        slope_map = {"Upsloping":0,  "Flat":1, "Downsloping":2}
        slope_ip  =str.selectbox("Slope of the peak exercise ST segment",["Upsloping","Flat","Downsloping"])
        slope = slope_map[slope_ip]
        thal_map = {"Normal":0,"Fixed defect":1,"Reversible defect":2,"Not described":3}
        thal_ip=str.selectbox("Thalium stress test result",["Normal","Fixed defect","Reversible defect","Not described"])
        thal = thal_map[thal_ip]
    with col2:
        sex_mapping={"Male": 0, "Female": 1}
        sex_input = str.selectbox("Sex",["Male","Female"])
        sex = sex_mapping[sex_input]
        trestbps = str.text_input("Resting Blood Pressure",placeholder="in mm Hg")
        fbs_map = {"True":1,"False":0}
        fbs_ip=str.selectbox("Fasting Blood Sugar > 120 mg/dl",["True","False"])
        fbs = fbs_map[fbs_ip]
        thalach = str.text_input("Maximum Heart Rate")
        oldpeak = str.text_input("ST depression induced")
        ca = str.text_input("Number of major vessels (0-4) colored by fluoroscopy")


    # Upload CSV file
    uploaded_file = str.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        str.write(df)  # Display the uploaded data

        # Extract numeric features from the uploaded dataset (assuming column names match)
        age = df['age'].iloc[0] if 'age' in df.columns else 0
        sex = df['sex'].iloc[0] if 'sex' in df.columns else 0
        cp = df['cp'].iloc[0] if 'cp' in df.columns else 0
        trestbps = df['trestbps'].iloc[0] if 'trestbps' in df.columns else 0
        chol = df['chol'].iloc[0] if 'chol' in df.columns else 0
        fbs = df['fbs'].iloc[0] if 'fbs' in df.columns else 0
        restecg = df['restecg'].iloc[0] if 'restecg' in df.columns else 0
        thalach = df['thalach'].iloc[0] if 'thalach' in df.columns else 0
        exang = df['exang'].iloc[0] if 'exang' in df.columns else 0
        oldpeak = df['oldpeak'].iloc[0] if 'oldpeak' in df.columns else 0
        slope = df['slope'].iloc[0] if 'slope' in df.columns else 0
        ca = df['ca'].iloc[0] if 'ca' in df.columns else 0
        thal = df['thal'].iloc[0] if 'thal' in df.columns else 0
    # Prediction
    heart_diagnose = ""

    if str.button("Heart Disease Test Result"):
        heart_predict = heart_disease_model.predict([[int(age),int(sex),int(cp),int(trestbps),int(chol),int(fbs),int(restecg),int(thalach),int(exang),float(oldpeak),int(slope),int(ca),int(thal)]])
        if(heart_predict[0] == 1):
            heart_diagnose = "The person has heart disease"
        else:
            heart_diagnose = "The person doesn't have heart disease"
        str.success(heart_diagnose)
        if heart_predict[0]==1:
            treatment_plan = heart_disease_treatment_plan(...)
            str.write(treatment_plan)  # Display the treatment plan
            str.markdown(get_download_link(treatment_plan), unsafe_allow_html=True)

elif(select == 'Diabetes'):
    str.title("Diabetes Prediction")

    #str.image('D:/Minor Project/multiple_disease/360_F_507662376_BTKmPlIGBvKlRHWKRNeFt7bj7H2SynQm.jpg', caption='Your Image', use_column_width=True,)
    #User Input:
    col1,col2 = str.columns(2)
    with col1:
        preg = str.text_input("Pregnancies", placeholder="No. of times pregnant")
        glu = str.text_input("Glucose", placeholder="Plasma glucose concentration")
        bp = str.text_input("Blood Pressure", placeholder="Diastolic BP(mm Hg)")
        st = str.text_input("Skin Thickness", placeholder="Triceps skin fold (mm)")
    with col2:
        ins = str.text_input("Insulin", placeholder="2-Hour serum insulin(muU/ml)")
        bmi = str.text_input("BMI", placeholder="Body Mass Index")
        dbf = str.text_input("Diabetes Pedigree Function")
        age = str.text_input("Age")

    uploaded_file = str.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        str.write(df)  # Display the uploaded data
        
        # Assuming the column names in the CSV file match the input fields
        preg = df['Pregnancies'].iloc[0] if 'Pregnancies' in df.columns else 0
        glu = df['Glucose'].iloc[0] if 'Glucose' in df.columns else 0
        bp = df['BloodPressure'].iloc[0] if 'BloodPressure' in df.columns else 0
        st = df['SkinThickness'].iloc[0] if 'SkinThickness' in df.columns else 0
        ins = df['Insulin'].iloc[0] if 'Insulin' in df.columns else 0
        bmi = df['BMI'].iloc[0] if 'BMI' in df.columns else 0
        dbf = df['DiabetesPedigreeFunction'].iloc[0] if 'DiabetesPedigreeFunction' in df.columns else 0
        age = df['Age'].iloc[0] if 'Age' in df.columns else 0

    # Prediction
    diabetes_diagnose = ""
    if str.button("Diabetes Test Result"):
        diabetes_pred  = diabetes_model.predict([[int(preg),int(glu),int(bp),int(st),int(ins),float(bmi),float(dbf),int(age)]])
        if(diabetes_pred[0]==1):
            diabetes_diagnose = "POSITIVE: The person has diabetes"
        else:
            diabetes_diagnose = "NEGATIVE: The person doesn't have diabetes"
        str.success(diabetes_diagnose)
        if diabetes_pred[0]==1:
            treatment_plan = diabetes_treatment_plan(...)
            str.write(treatment_plan)  # Display the treatment plan
            str.markdown(get_download_link(treatment_plan), unsafe_allow_html=True)

    


elif(select == 'Parkinson'):
    str.title("Parkinson Prediction")

    #User Input Fields:
    col1, col2 = str.columns(2)
    with col1:
        fo = str.text_input("Fundamental Frequency (F0)", placeholder="Average vocal cord oscillation rate")
        flo = str.text_input("Lowest F0", placeholder="Lowest vocal cord oscillation rate")
        jitter = str.text_input("Jitter", placeholder="Variation in vocal cord oscillation")
        jitter_abs = str.text_input("Jitter Abs", placeholder="Absolute jitter")
        rap = str.text_input("RAP", placeholder="Relative amplitude perturbation")
        ppq = str.text_input("PPQ", placeholder="Period perturbation quotient")
        ddp = str.text_input("DDP", placeholder="Degree of perturbation")
        apq3 = str.text_input("APQ3", placeholder="Amplitude perturbation quotient")
        apq5 = str.text_input("APQ5", placeholder="Amplitude perturbation quotient")
        apq = str.text_input("APQ", placeholder="Amplitude perturbation quotient")
        d2 = str.text_input("D2", placeholder="Nonlinear measure of fundamental")
    with col2:
        fhi = str.text_input("Highest F0", placeholder="Highest vocal cord oscillation rate")
        shimmer = str.text_input("Shimmer", placeholder="Cycle-to-cycle amplitude variation")
        shimmer_db = str.text_input("Shimmer dB", placeholder="Shimmer in decibels")
        dda = str.text_input("DDA", placeholder="Degree of voice breaks")
        nhr = str.text_input("NHR", placeholder="Noise-to-harmonics ratio")
        hnr = str.text_input("HNR", placeholder="Harmonics-to-noise ratio")
        rpde = str.text_input("RPDE", placeholder="Recurrence period density entropy")
        dfa = str.text_input("DFA", placeholder="Detrended fluctuation analysis")
        sp1 = str.text_input("SP1", placeholder="Nonlinear measure of fundamental")
        sp2 = str.text_input("SP2", placeholder="Nonlinear measure of fundamental")
        ppe = str.text_input("PPE", placeholder="Pitch period entropy")
    
    uploaded_file = str.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        str.write(df)  # Display the uploaded data
    
    # Assuming the column names in the CSV file match the input fields
        fo = df['Fundamental Frequency (F0)'].iloc[0] if 'Fundamental Frequency (F0)' in df.columns else 0
        fhi = df['Highest F0'].iloc[0] if 'Highest F0' in df.columns else 0
        flo = df['Lowest F0'].iloc[0] if 'Lowest F0' in df.columns else 0
        jitter = df['Jitter'].iloc[0] if 'Jitter' in df.columns else 0
        jitter_abs = df['Jitter Abs'].iloc[0] if 'Jitter Abs' in df.columns else 0
        rap = df['RAP'].iloc[0] if 'RAP' in df.columns else 0
        ppq = df['PPQ'].iloc[0] if 'PPQ' in df.columns else 0
        ddp = df['DDP'].iloc[0] if 'DDP' in df.columns else 0
        shimmer = df['Shimmer'].iloc[0] if 'Shimmer' in df.columns else 0
        shimmer_db = df['Shimmer dB'].iloc[0] if 'Shimmer dB' in df.columns else 0
        apq3 = df['APQ3'].iloc[0] if 'APQ3' in df.columns else 0
        apq5 = df['APQ5'].iloc[0] if 'APQ5' in df.columns else 0
        apq = df['APQ'].iloc[0] if 'APQ' in df.columns else 0
        dda = df['DDA'].iloc[0] if 'DDA' in df.columns else 0
        nhr = df['NHR'].iloc[0] if 'NHR' in df.columns else 0
        hnr = df['HNR'].iloc[0] if 'HNR' in df.columns else 0
        rpde = df['RPDE'].iloc[0] if 'RPDE' in df.columns else 0
        dfa = df['DFA'].iloc[0] if 'DFA' in df.columns else 0
        sp1 = df['SP1'].iloc[0] if 'SP1' in df.columns else 0
        sp2 = df['SP2'].iloc[0] if 'SP2' in df.columns else 0
        d2 = df['D2'].iloc[0] if 'D2' in df.columns else 0
        ppe = df['PPE'].iloc[0] if 'PPE' in df.columns else 0

    #Prediction
    parkinson_diagnose=""
    if str.button("Parkinson's Disease Test Result"):
        park_predict = parkinson_disease_model.predict([[float(fo),float(fhi),float(flo),float(jitter),float(jitter_abs),float(rap),float(ppq),float(ddp),float(shimmer),float(shimmer_db),float(apq3),float(apq5),float(apq),float(dda),float(nhr),float(hnr),float(rpde),float(dfa),float(sp1),float(sp2),float(d2),float(ppe)]])
        if(park_predict[0]==1):
            parkinson_diagnose="The person has Parkinson's Disease"
        else:
            parkinson_diagnose="The person doesn't have Parkinson's Disease"
        str.success(parkinson_diagnose)
        if park_predict[0]==1:
            treatment_plan = parkinson_disease_treatment_plan(...)
            str.write(treatment_plan)  # Display the treatment plan
            str.markdown(get_download_link(treatment_plan), unsafe_allow_html=True)


elif(select == 'Cancer'):
    str.title("Cancer Prediction")

    #User Input
    col1, col2 = str.columns(2)
    with col1:
        radius_mean = str.text_input("Radius Mean", placeholder="Mean of distances from center to points on the perimeter")
        perimeter_mean = str.text_input("Perimeter Mean", placeholder="Mean size of the perimeter")
        smoothness_mean = str.text_input("Smoothness Mean", placeholder="Mean of local variation in radius lengths")
        concavity_mean = str.text_input("Concavity Mean", placeholder="Mean severity of concave portions of the contour")
        symmetry_mean = str.text_input("Symmetry Mean", placeholder="Mean symmetry of the cell nuclei")
        radius_se = str.text_input("Radius SE", placeholder="Standard error of the mean of distances")
        perimeter_se = str.text_input("Perimeter SE", placeholder="Standard error of the perimeter")
        smoothness_se  = str.text_input("Smoothness SE")
        concavity_se = str.text_input("Concavity SE", placeholder="Standard error of severity of concave portions")
        symmetry_se = str.text_input("Symmetry SE", placeholder="Standard error for symmetry")
        radius_worst = str.text_input("Radius Worst", placeholder="Worst radius")
        perimeter_worst = str.text_input("Perimeter Worst")
        smoothness_worst = str.text_input("Smoothness Worst", placeholder="Worst smoothness")
        concavity_worst = str.text_input("Concavity Worst", placeholder="Worst severity of concave portions")
        symmetry_worst = str.text_input("Symmetry Worst", placeholder="Worst symmetry")
    with col2:
        texture_mean = str.text_input("Texture Mean", placeholder="Standard deviation of gray-scale values")
        area_mean = str.text_input("Area Mean", placeholder="Mean area of the nucleus")
        compactness_mean = str.text_input("Compactness Mean", placeholder="Mean perimeter^2 / area - 1.0")
        concave_points_mean = str.text_input("Concave Points Mean", placeholder="Mean for number of concave portions")
        fractal_dimension_mean = str.text_input("Fractal Dimension Mean", placeholder="Mean for 'coastline approximation'")
        texture_se = str.text_input("Texture SE", placeholder="Standard error for gray-scale values")
        area_se = str.text_input("Area SE", placeholder="Standard error for area")
        compactness_se = str.text_input("Compactness SE")
        concave_points_se = str.text_input("Concave points SE")
        fractal_dimension_se = str.text_input("Fractal Dimension SE", placeholder="Standard error of complexity")
        texture_worst = str.text_input("Texture Worst", placeholder="Worst texture")
        area_worst = str.text_input("Area Worst", placeholder="Worst area")
        compactness_worst = str.text_input("Compactness Worst", placeholder="Worst compactness")
        concave_points_worst = str.text_input("Concave Points Worst", placeholder="Worst number of concave portions")
        fractal_dimension_worst = str.text_input("Fractal Dimension Worst", placeholder="Worst fractal dimension")
        uploaded_file = str.file_uploader("Upload CSV file", type=['csv'])
    if  uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        str.write(df)  # Display the uploaded data
    
        # Assuming the column names in the CSV file match the input fields
        radius_mean = df['radius_mean'].iloc[0] if 'radius_mean' in df.columns else 0
        perimeter_mean = df['perimeter_mean'].iloc[0] if 'perimeter_mean' in df.columns else 0
        smoothness_mean = df['smoothness_mean'].iloc[0] if 'smoothness_mean' in df.columns else 0
        concavity_mean = df['concavity_mean'].iloc[0] if 'concavity_mean' in df.columns else 0
        symmetry_mean = df['symmetry_mean'].iloc[0] if 'symmetry_mean' in df.columns else 0
        radius_se = df['radius_se'].iloc[0] if 'radius_se' in df.columns else 0
        perimeter_se = df['perimeter_se'].iloc[0] if 'perimeter_se' in df.columns else 0
        smoothness_se = df['smoothness_se'].iloc[0] if 'smoothness_se' in df.columns else 0
        concavity_se = df['concavity_se'].iloc[0] if 'concavity_se' in df.columns else 0
        symmetry_se = df['symmetry_se'].iloc[0] if 'symmetry_se' in df.columns else 0
        radius_worst = df['radius_worst'].iloc[0] if 'radius_worst' in df.columns else 0
        perimeter_worst = df['perimeter_worst'].iloc[0] if 'perimeter_worst' in df.columns else 0
        smoothness_worst = df['smoothness_worst'].iloc[0] if 'smoothness_worst' in df.columns else 0
        concavity_worst = df['concavity_worst'].iloc[0] if 'concavity_worst' in df.columns else 0
        symmetry_worst = df['symmetry_worst'].iloc[0] if 'symmetry_worst' in df.columns else 0
        texture_mean = df['texture_mean'].iloc[0] if 'texture_mean' in df.columns else 0
        area_mean = df['area_mean'].iloc[0] if 'area_mean' in df.columns else 0
        compactness_mean = df['compactness_mean'].iloc[0] if 'compactness_mean' in df.columns else 0
        concave_points_mean = df['concave_points_mean'].iloc[0] if 'concave_points_mean' in df.columns else 0
        fractal_dimension_mean = df['fractal_dimension_mean'].iloc[0] if 'fractal_dimension_mean' in df.columns else 0
        texture_se = df['texture_se'].iloc[0] if 'texture_se' in df.columns else 0
        area_se = df['area_se'].iloc[0] if 'area_se' in df.columns else 0
        compactness_se = df['compactness_se'].iloc[0] if 'compactness_se' in df.columns else 0
        concave_points_se = df['concave_points_se'].iloc[0] if 'concave_points_se' in df.columns else 0
        fractal_dimension_se = df['fractal_dimension_se'].iloc[0] if 'fractal_dimension_se' in df.columns else 0
        texture_worst = df['texture_worst'].iloc[0] if 'texture_worst' in df.columns else 0
        area_worst = df['area_worst'].iloc[0] if 'area_worst' in df.columns else 0
        compactness_worst = df['compactness_worst'].iloc[0] if 'compactness_worst' in df.columns else 0
        concave_points_worst = df['concave_points_worst'].iloc[0] if 'concave_points_worst' in df.columns else 0
        fractal_dimension_worst = df['fractal_dimension_worst'].iloc[0] if 'fractal_dimension_worst' in df.columns else 0


    #Prediction
    cancer_diagnose=""
    if str.button("Cancer Test Result"):
        #cancer_predict = cancer_model.predict([[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]])
        cancer_predict = cancer_model.predict([[float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), float(smoothness_mean),
                                                float(compactness_mean), float(concavity_mean), float(concave_points_mean), float(symmetry_mean),
                                                 float(fractal_dimension_mean), float(radius_se), float(texture_se), float(perimeter_se), float(area_se),
                                                 float(smoothness_se), float(compactness_se), float(concavity_se), float(concave_points_se), float(symmetry_se),
                                                 float(fractal_dimension_se), float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst),
                                                 float(smoothness_worst), float(compactness_worst), float(concavity_worst), float(concave_points_worst),
                                                 float(symmetry_worst), float(fractal_dimension_worst)]])
        if(cancer_predict[0]==1):
            cancer_diagnose="The person has Malignant Cancer"
        else:
            cancer_diagnose="The person has Benign Cancer"
        str.success(cancer_diagnose)
        if cancer_predict[0]==1:
            treatment_plan = cancer_treatment_plan(...)
            str.write(treatment_plan)  # Display the treatment plan
            str.markdown(get_download_link(treatment_plan), unsafe_allow_html=True)

elif select == 'Brain Tumor':
    str.title('Brain Tumor Prediction')
    tumor_map={0:"Glinoma tumor",1:"Meningioma Tumor",2:"No Tumor",3:"Pituitary Tumor"}
    #User Input
    tumor_diagnose = ""
    uploaded_file = str.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        str.image(image, caption='Uploaded Image',use_column_width=True)
        if str.button("Brain Tumor Test Result"):
            # Preprocess the image
            img_arr = np.array(image)
            resized_img = cv2.resize(img_arr,(150,150))
            img_arr = resized_img.reshape(1,150,150,3)
            # Make prediction
            prediction = brain_tumor_model.predict(img_arr)
            res = prediction.argmax()
            tumor_diagnose = "The person has "+ tumor_map[res]
            str.success(tumor_diagnose)
            if res!=2:
                str.write("Treatment Plan:")
                str.write("1. Surgery: The primary treatment for many brain tumors involves surgical removal of the tumor. The goal is to remove as much of the tumor as possible while preserving neurological function.")
                str.write("2. Radiation therapy: Radiation therapy uses high-energy beams to kill cancer cells. It is often used after surgery to kill any remaining cancer cells or as a primary treatment for tumors that are difficult to remove surgically.")
                str.write("3. Chemotherapy: Chemotherapy uses drugs to kill cancer cells. It is often used in combination with surgery and/or radiation therapy, especially for aggressive or recurrent tumors.")
                str.write("4. Steroids: Steroids may be prescribed to reduce swelling and relieve symptoms such as headaches and edema (swelling) around the tumor.")





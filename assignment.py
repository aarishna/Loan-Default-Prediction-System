"""
Code:   Loan Default Predictor & Circular DLL
Author: Aarish Naiyer
When:   April 7rd, 2025
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from carousel import Carousel, DLinkedListNode
scaler=StandardScaler()

def readfile(filename,datafile): #function reads file and stores the data as a list of lists
    with open(filename) as file:
        lines=file.readlines()  # stores as list of lists
        for i in lines:
            data=i.strip().split(',')
            datafile.append(data)

        
def fixing_file(clean_file, datafile):
    
    for i in range(3):
        print()
    headers = datafile[0]  # Get column names
    num_cols = len(headers)
    missing_counts = [0] * num_cols

    # Count missing values by column
    for row in datafile[1:]:
        for i, value in enumerate(row):
            if value.strip() == "":
                missing_counts[i] += 1

    # Print initial number of rows
    print(f"Initial number of rows: {len(datafile) - 1}")  # excluding header
    print()

    # Print which columns have missing values
    for i, count in enumerate(missing_counts):
        if count > 0:
            print(f"Column {headers[i]}: {count} values missing")
            print()

    clean_file.append(headers)

    # Adding only rows with NO missing values AND age < 90
    for row in datafile[1:]:
        if '' not in row and int(row[0]) < 90:
            clean_file.append(row)

    # Final output
    print(f"Remaining number of rows: {len(clean_file) - 1}")
    

    
    
    
    
      
def histogram(clean_file):          #function to print out histograms using the matplotlib library
    bins=list(range(0,100,10))      #creating the range of x axis being in the groups of 10 up to 100
    default_data=[int(row[0]) for row in clean_file[1:] if row[8]=='1']
    plt.hist(default_data,bins=bins,edgecolor='black')
    plt.title("Loans in Default")
    plt.xlabel('Age (in years)')
    plt.ylabel("No. of borrowers")
    plt.show()
    not_default_data=[int(row[0]) for row in clean_file[1:] if row[8]=='0']
    plt.hist(not_default_data,bins=bins,edgecolor='black')
    plt.title("Loans in Not default")
    plt.xlabel('Age (in years)')
    plt.ylabel("No. of borrowers")
    plt.show()



def pie_chart(clean_file):          #function to print out a pie chart using the matplotlib library
    defaulted=0     #count of defaulted homeowners
    not_defaulted=0     #count of not-defaulted homeowners
    for row in clean_file:
        if row[2]=="OWN": 
            if row[8]=='1':
                defaulted+=1
            if row[8]=='0':
                not_defaulted+=1
    default_borrowers=0     #count of defaulted borrowers 
    non_default_borrowers=0     #count of not-defaulted borrowers
    for row in clean_file:
        
        if row[8]=='1':
            default_borrowers+=1
        if row[8]=='0':
            non_default_borrowers+=1

            
    percentage_defaulted=defaulted*100/len(clean_file)      #calculating the percentage manually    
    percentage_not_defaulted=not_defaulted*100/len(clean_file)      #calculating the percentage manually
    plt.figure(figsize=(8,8))
    labels=['Defaulted','Not Defaulted']
    percentages=[percentage_defaulted,percentage_not_defaulted]
    plt.pie(percentages,labels=labels,autopct='%1.1f%%',startangle=140)
    plt.title("Homeowners: Default vs Not-Default")
    plt.show()
    for i in range(4):
        print()
    print("Number of default borrowers:", default_borrowers)
    print()
    print("Number of not default borrowers:", non_default_borrowers)
    for i in range(5):
        print()


def scaling_data(clean_file,loan_file):
    clean_file_num=[[int(row[6]),int(row[1])] for row in clean_file[1:]]
    scaled_data=scaler.fit_transform(clean_file_num)
    loan_testing_data=[[int(row[7]),int(row[1])] for row in loan_file[1:]]
    scaled_loan_data=scaler.fit_transform(loan_testing_data)
    return scaled_data,scaled_loan_data
        
def training_data(clean_file,scaled_train_data):
    x_train=[]
    y_train=[int(clean_file[i+1][8]) for i in range(len(scaled_train_data))]
    for i in range(len(scaled_train_data)):
        scaled_values=scaled_train_data[i]
        credit_history=int(clean_file[i+1][11])
        features=[scaled_values[0],scaled_values[1],credit_history]
        x_train.append(features)
        

    clf = DecisionTreeClassifier(random_state=42)  # Create the model
    clf.fit(x_train, y_train)  # Train the model with the data
    return x_train,y_train
    
    
def test_data(x_train,y_train,clean_file,scaled_test_data):
    x_test=[]
    y_test=[int(clean_file[i+1][8]) for i in range(len(scaled_test_data))]
    for i in range(len(scaled_test_data)):
        scaled_values=scaled_test_data[i]
        credit_history=int(clean_file[i+1][11])
        features=[scaled_values[0],scaled_values[1],credit_history]
        x_test.append(features)
        
    clf = DecisionTreeClassifier(random_state=42)  # Create the model
    clf.fit(x_train,y_train)
    # Get predictions from the trained model
    y_pred = clf.predict(x_test)
    # Print evaluation metrics
    print(f"Test Accuracy:{accuracy_score(y_test, y_pred):.2f}")  # Accuracy
    print(classification_report(y_test, y_pred))  # Precision, Recall, F1-score
    print()
    print(confusion_matrix(y_test, y_pred))  # Confusion Matrix
    for i in range(5):
        print()



def loan_testing(scaled_loan_data,loan_file):
    loan_test_data=[]
    for i in range (len(loan_file)-1):
        loan_values=scaled_loan_data[i]
        loan_test_credit_history=int(loan_file[i+1][11])
        loan_features=[loan_values[0],loan_values[1],loan_test_credit_history]
        loan_test_data.append(loan_features)
    return loan_test_data

def predicting_loan(loan_test_data,x_train,y_train):
    clf=DecisionTreeClassifier(random_state=42)
    clf.fit(x_train,y_train)
    loan_pred=clf.predict(loan_test_data)
    return loan_pred

def making_carousel(loan_file,loan_pred,applicants):
    headers = loan_file[0]  # header row

    for row,prediction in zip(loan_file[1:],loan_pred):
        
        applicant_dict = dict(zip(headers, row))
        applicant_dict["loan_status"]=prediction
        applicants.add(applicant_dict)
        
        
    # print(applicants)
    return applicants
        


def UI(applicants):
    applicant = applicants.getCurrentData()
    applicants.moveNext()
    while True:
        applicant = applicants.getCurrentData()
        print("--------------------------------------------------")
        print(f"Borrower: {applicant['borrower']}")
        print(f"\nAge: {applicant['person_age']}")
        print(f"\nIncome: ${applicant['person_income']}")
        print(f"\nHome ownership: {applicant['person_home_ownership']}")
        print(f"\nEmployment: {applicant['person_emp_length']}")
        print(f"\nLoan intent: {applicant['loan_intent']}")
        print(f"\nLoan grade: {applicant['loan_grade']}")
        print(f"\nAmount: ${applicant['loan_amnt']}")
        print(f"\nInterest Rate: {applicant['loan_int_rate']}")
        print(f"\nLoan percent income: {applicant['loan_percent_income']}")
        print(f"\nHistorical Defaults: {'Yes' if applicant['cb_person_default_on_file'] == 'Y' else 'No'}")
        print(f"\nCredit History: {applicant['cb_person_cred_hist_length']} years")
        print("\n--------------------------------------------------")

        if applicant["loan_status"] == 1:
            print("\nPredicted loan status: Will default")
            print("Recommend: \033[38;5;196mReject\033[0m")
        else:
            print("\nPredicted loan status: Will not default")
            print("Recommend: \033[32mAccept\033[0m")

        print("\n--------------------------------------------------")
        print("\nOptions:")
        print("1 → Next applicant")
        print("2 → Previous applicant")
        print("0 → Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            applicants.moveNext()
        elif choice == "2":
            applicants.movePrevious()
        elif choice == "0":
            print()
            print()
            print("BYEBYE")
            print()
            return
        else:
            print("Invalid input. Please enter 1, 2, or 0.")

    
    
    
def main():
    applicants=Carousel()
    print("==========PART ONE==========")
    data=[]
    clean_file=[]
    loan_file=[]
    test_file=[]
    readfile("credit_risk_train.csv",datafile=data)
    readfile("credit_risk_test.csv",datafile=test_file)
    readfile("loan_requests.csv",datafile=loan_file)

    fixing_file(clean_file=clean_file,datafile=data)
    histogram(clean_file=clean_file)
    pie_chart(clean_file=clean_file)
    scaled_data,scaled_loan_data = scaling_data(clean_file,loan_file)
    # training_data(clean_file,scaled_data)
    x_train,y_train=training_data(clean_file,scaled_data)
    
    scaled_test_data,scaled_loan_data=scaling_data(test_file,loan_file)
    test_data(x_train=x_train,y_train=y_train,clean_file=test_file,scaled_test_data=scaled_test_data)
    
    loan_test_data=loan_testing(scaled_loan_data,loan_file)
    loan_pred=predicting_loan(loan_test_data,x_train,y_train)
    
    
    input("Press'Enter' to go to the Next part:")
    
    
    
    making_carousel(loan_file,loan_pred,applicants)
    
    UI(applicants)
if __name__=="__main__":
    main()

import pandas as pd


df = pd.read_csv('hospital_patients_real_world.csv')

df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])

df['Diagnosis'] = df['Diagnosis'].fillna('Unknown')

df['LengthOfStay'] = (df['DischargeDate'] - df['AdmissionDate']).dt.days

clean_df = df[df['LengthOfStay'] >= 0].copy()

top_10_diagnosis = clean_df['Diagnosis'].value_counts().head(10)
avg_age = clean_df['Age'].mean()
avg_stay = clean_df['LengthOfStay'].mean()


print(f"Total valid records analyzed: {len(clean_df)}")
print(f"Average Patient Age: {avg_age:.1f} years")
print(f"Average Hospital Stay: {avg_stay:.1f} days")
print("\nTop 5 Diagnoses:")
print(top_10_diagnosis.head(5))

clean_df.to_csv('Cleaned_Hospital_Data.csv', index=False)
print("\nSuccess: 'Cleaned_Hospital_Data.csv' has been generated.")
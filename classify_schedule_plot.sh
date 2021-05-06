echo "Predicting Labels for TestingData.txt."
python3 classify.py

echo "Scheduling and plotting abnormal testing data."
python3 schedule_plot.py

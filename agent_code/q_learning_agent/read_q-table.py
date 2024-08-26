import pickle
import matplotlib.pyplot as plt

# Laden der Q-Tabelle
with open('q_table.pkl', 'rb') as file:
    q_table = pickle.load(file)

# Extrahiere alle Q-Werte
all_q_values = [q for q_values in q_table.values() for q in q_values]

# Plotten der Verteilung der Q-Werte
plt.figure(figsize=(10, 6))
plt.hist(all_q_values, bins=50, color='blue', alpha=0.7)
plt.title("Verteilung der Q-Werte")
plt.xlabel("Q-Wert")
plt.ylabel("Häufigkeit")
plt.show()

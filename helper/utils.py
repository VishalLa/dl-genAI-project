from pathlib import Path
import pandas as pd

def get_fold_split(meta_path: Path, fold_number: int):
    df = pd.read_csv(meta_path)

    train_df = df[df["fold"] != fold_number]
    test_df  = df[df["fold"] == fold_number]

    return train_df, test_df

# NUM_FOLDS = 5
# all_fold_accuracies = []

# for fold in range(1, NUM_FOLDS+1):

#     print(f"\n========= FOLD {fold} =========")

#     train_df, test_df = get_fold_split("ESC-50/meta/esc50.csv", fold)

#     train_dataset = ESC50Dataset(train_df, "ESC-50/audio")
#     test_dataset  = ESC50Dataset(test_df,  "ESC-50/audio")

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

#     model = AudioCNN()  # (we will build next)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = torch.nn.CrossEntropyLoss()

#     train_model(model, train_loader, optimizer, criterion)
#     acc = evaluate_model(model, test_loader)

#     print("Fold Accuracy:", acc)
#     all_fold_accuracies.append(acc)


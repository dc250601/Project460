import torch
from tqdm.auto import tqdm
import gc
import wandb
import augment
import torchvision.datasets as datasets
import VICReg
import loss as Loss

transform = augment.TrainTransform()
dataset = datasets.ImageFolder("../Data_Extraction/data/",transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=256,
                                            shuffle=True,
                                            drop_last = True,
                                            num_workers=16,
                                            pin_memory = True)


model = VICReg.VICReg()


optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = Loss.VicregLoss(batch_size=256)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True,threshold = 0.01,patience = 3, factor = 0.5)

model = model.to("cuda:3")



wandb.login(key="cb53927c12bd57a0d943d2dedf7881cfcdcc8f09")
wandb.init(
    project = "460-VICReg",
    name = "Try3"
)


scaler = torch.cuda.amp.GradScaler()
#--------------------------
wandb.watch(model, log_freq=50)
#---------------------------
w_intr = 50

for epoch in range(0,100,1):
    train_loss = 0
    train_steps = 0
    val_loss_a = 0
    val_loss_b = 0
    val_loss_c = 0
    val_steps = 0

    model.train()
    for (x,y), _ in tqdm(dataloader):
        x = x.to("cuda:3")
        y = y.to("cuda:3")

        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        with torch.cuda.amp.autocast():
            output_x,output_y = model(x,y)
            loss = criterion(output_x,output_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_steps += 1

# ---------------------------------------------------

    model.eval()
    with torch.no_grad():
        for (x,y), _ in tqdm(dataloader):
            x = x.to("cuda:3")
            y = y.to("cuda:3")
            output_x,output_y = model(x,y)
            loss_a, loss_b, loss_c = criterion(x=output_x,y=output_y,mode="eval")

            val_loss_a += loss_a.item()
            val_loss_b += loss_b.item()
            val_loss_c += loss_c.item()
            val_steps += 1

            if val_steps > 0.25*train_steps:
                break

    train_loss = train_loss/train_steps
    val_loss_a = val_loss_a/val_steps
    val_loss_b = val_loss_b/val_steps
    val_loss_c = val_loss_c/val_steps
    print("----------------------------------------------------")
    print("Epoch No" , epoch)
    print("The Loss ",train_loss)
    print("----------------------------------------------------")

    PATH = "weight.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
                 }, PATH)
    scheduler.step(train_loss)
    curr_lr = scheduler._last_lr[0]
    wandb.log({"Epoch": epoch,
                "Epoch_loss": train_loss,
                " val_repr_loss":val_loss_a,
                " val_std_loss":val_loss_b,
                "val_cov_loss":val_loss_c,
                "Lr": curr_lr}
                )

    gc.collect()

wandb.finish()

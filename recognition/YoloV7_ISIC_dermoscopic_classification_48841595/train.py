import modules



train_loader, test_loader = get_loaders(
    train_img_dir, train_label_dir,
    val_img_dir, val_label_dir,  # use test dir here
    batch_size=batch_size,
    img_size=img_size
)

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch in progress:
        imgs = batch['image'].to(device)
        bboxes = batch['bboxes'].to(device)

        optimizer.zero_grad()

        # Convert bboxes to YOLO target format
        targets = []
        for i, boxes in enumerate(bboxes):
            if boxes.numel() == 0:
                continue
            img_idx = torch.full((boxes.shape[0],1), i, device=device)
            boxes_for_yolo = torch.cat([img_idx, boxes], dim=1)
            targets.append(boxes_for_yolo)
        targets = torch.cat(targets, dim=0) if targets else torch.zeros((0,6), device=device)

        preds = model(imgs)

        # Compute loss
        loss = preds['loss'] if isinstance(preds, dict) and 'loss' in preds else torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            bboxes = batch['bboxes'].to(device)

            # Targets
            targets = []
            for i, boxes in enumerate(bboxes):
                if boxes.numel() == 0:
                    continue
                img_idx = torch.full((boxes.shape[0],1), i, device=device)
                boxes_for_yolo = torch.cat([img_idx, boxes], dim=1)
                targets.append(boxes_for_yolo)
            targets = torch.cat(targets, dim=0) if targets else torch.zeros((0,6), device=device)

            preds = model(imgs)
            loss = preds['loss'] if isinstance(preds, dict) and 'loss' in preds else torch.tensor(0.0, device=device)
            test_loss += loss.item()
            test_acc += compute_accuracy(preds['pred'] if 'pred' in preds else torch.zeros_like(targets), targets)

    val_losses.append(test_loss / len(test_loader))
    val_accuracies.append(test_acc / len(test_loader))
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Test Loss: {val_losses[-1]:.4f} | Test Acc: {val_accuracies[-1]:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, f"yolov7_epoch{epoch+1}.pt"))
    model.train()


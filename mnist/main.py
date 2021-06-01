import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

import torchvision as tv

from time import time
from src.model import Model
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from src.argument import parser, print_args

def compute_all_layer_margin(self, model, data, label):
    rem = self.attack.epsilon

    l = 0
    r = 0.5
    for t in range(8):
        self.attack.epsilon = (l + r) / 2
        output = torch.max(model(self.attack.perturb(data, label, 'mean', True)), dim=1)[1]

        # print(output.item(), label.item())
        if (output.item() == label.item()):
            r = (l + r) / 2
        else:
            l = (l + r) / 2
    return (l + r) / 2

    self.attack.epsilon = rem

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack
        

     
    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)




    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.Adam(model.parameters(), args.learning_rate)

        _iter = 0

        begin_time = time()

        for epoch in range(1, args.max_epoch+1):

            output_margin_test = 0
            dataset_size = 0
            all_layer_margin_test = 0
            train_acc = 0

            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)


                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)

                pred = torch.max(output, dim=1)[1]
                train_acc += evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

                for i in range(label.size()[0]):
                    # cx = data[i].clone()
                    # cy = label[i]
                    # cx = cx.unsqueeze(dim = 0)
                    # cy = cy.unsqueeze(dim = 0)
                    # print(data.size())
                    # print(cx.size())
                    # all_layer_margin_test += compute_all_layer_margin(self, model, cx, cy)

                    x = output[i].clone()
                    y = label[i].item()
                    x = F.softmax(x, dim = 0)
                    val_l = x[y].item()
                    x[y] = 0
                    val_other = torch.max(x).item()
                    dataset_size += 1
                    output_margin_test += max(0, val_l - val_other)
                    # faq = model(x)


                opt.zero_grad()
                loss.backward()
                opt.step()

                # if _iter % args.n_eval_step == 0:

                    # if adv_train:
                        # with torch.no_grad():
                            # stand_output = model(data, _eval=True)
                        # pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        # std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        # pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        # adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
 
                    # else:
                        # adv_data = self.attack.perturb(data, label, 'mean', False)

                        # with torch.no_grad():
                            # adv_output = model(adv_data, _eval=True)
                        # pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        # adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100 
                        # pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        # std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    # only calculating the training time


                # if _iter % args.n_store_image_step == 0:
                    # tv.utils.save_image(torch.cat([data.cpu(), adv_data.cpu()], dim=0), 
                                        # os.path.join(args.log_folder, 'images_%d.jpg' % _iter), 
                                        # nrow=16)
                    

                _iter += 1
            

            file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % epoch)
            save_model(model, file_name)

            if va_loader is not None:
                va_acc, va_adv_acc, va_margin, va_layer = self.test(model, va_loader, True)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                logger.info('\n'+'='*20 +f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                    +'='*20)
                logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, train acc: {train_acc * 100.0 / dataset_size :.3f}% output margin in test: {output_margin_test / dataset_size}, output margin in val: {va_margin}, all-layer margin in test: {all_layer_margin_test / dataset_size}, all-layer margin in val: {va_layer}')
                logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

            begin_time = time()


    def test(self, model, loader, adv_test=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        total_margin = 0
        total_all_layer = 0
        cnt = 0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                for i in range(label.size()[0]):
                    # print(i)
                    if random.random() < 0:
                        cx = data[i].clone()
                        cy = label[i]
                        cx = cx.unsqueeze(dim = 0)
                        cy = cy.unsqueeze(dim = 0)
                        total_all_layer += compute_all_layer_margin(self, model, cx, cy)
                        cnt += 1

                    x = output[i].clone()
                    y = label[i].item()
                    x = F.softmax(x, dim = 0)
                    # print(sum(x))
                    # print(x, y)
                    val_l = x[y].item()
                    x[y] = 0
                    val_other = torch.max(x).item()
                    # print(val_other, val_l)
                        # print(output_margin_test)
                    total_margin += max(0, val_l - val_other)

                if adv_test:
                    # use predicted label as target label
                    # with torch.enable_grad():
                    adv_data = self.attack.perturb(data, pred, 'mean', False)

                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                    
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num , total_margin / num, total_all_layer / cnt

def main(args):

    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    model = Model(i_c=1, n_c=10)

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        tr_dataset = tv.datasets.MNIST(args.data_root, 
                                       train=True, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.MNIST(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        pass
    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
